import numpy as np
import torch
import torch.nn.functional as F
import os

from pathlib import Path

from agents.base_agent import Agent
from agents.algd.utils import soft_update
from agents.algd.VESDE import QNetwork, DiffusionPolicy, QcEnsemble


class ALGDAgent(Agent):
    def __init__(self, num_inputs, action_space, args):
        super().__init__()
        self.device = torch.device("cuda")
        self.discount = args.gamma
        self.safety_discount = args.safety_gamma
        self.critic_tau = args.tau
        self.critic_target_update_frequency = args.critic_target_update_frequency
        self.args = args

        self.update_counter = 0
        self.update_step = 0
        self.actor_update_step = 0

        self.rho = args.rho
        self.T = args.diffusion_steps_K
        
        # Score matching
        self.score_mc_samples = args.score_mc_samples
        self.score_beta = args.beta

        # Reward critic
        self.critic = QNetwork(num_inputs, action_space.shape[0], args.critic_hidden_size).to(self.device)
        self.critic_target = QNetwork(num_inputs, action_space.shape[0], args.critic_hidden_size).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Safety critics
        self.safety_critics = QcEnsemble(num_inputs, action_space.shape[0], args.qc_ens_size, args.critic_hidden_size).to(self.device)
        self.safety_critic_targets = QcEnsemble(num_inputs, action_space.shape[0], args.qc_ens_size, args.critic_hidden_size).to(self.device)
        self.safety_critic_targets.load_state_dict(self.safety_critics.state_dict())

        # Diffusion Policy
        self.policy = DiffusionPolicy(
            num_inputs=num_inputs,
            num_actions=action_space.shape[0],
            hidden_dim=args.score_model_hidden_dim,
            T=self.T,
            action_space=action_space
        ).to(self.device)
        
        # Lagrange multiplier for safety
        self.log_lam = torch.tensor(np.log(np.clip(0.6931, 1e-8, 1e8))).to(self.device)
        self.log_lam.requires_grad = True

        self.kappa = 0

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.policy.parameters(), lr=args.lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=args.lr)
        self.safety_critic_optimizer = torch.optim.Adam(self.safety_critics.parameters(), lr=args.qc_lr)
        self.log_lam_optimizer = torch.optim.Adam([self.log_lam], lr=args.lr)

        self.train()
        self.critic_target.train()
        self.safety_critic_targets.train()

        # Target cost
        if args.safetygym:
            self.target_cost = args.cost_lim * (1 - self.safety_discount ** args.epoch_length) / \
                               (1 - self.safety_discount) / args.epoch_length \
                               if self.safety_discount < 1 else args.cost_lim
        else:
            self.target_cost = args.cost_lim
        print("Constraint Budget: ", self.target_cost)
    
    @property
    def lam(self):
        return self.log_lam.exp()
    
    def get_last_log(self):
        return getattr(self, "last_log", {})
    
    def compute_LA(self, state, action):
        """
        Augmented Lagrangian:
        L_A(s,a,λ) = -min_j Q_j(s,a) + 1/(2 \rho) ( [λ + \rho * ( Qc̄(s,a) - h )]_+^2 - λ^2 )
        """
        Q1, Q2 = self.critic(state, action)
        Q_min = torch.min(Q1, Q2)

        QCs = self.safety_critics(state, action)
        qc_std, qc_mean = torch.std_mean(QCs, dim=0)
        if self.args.qc_ens_size == 1:
            qc_std = torch.zeros_like(qc_mean).to(self.device)
        qc_risk = qc_mean + 1.0 * qc_std

        lam = self.lam
        rho = self.rho
        h = self.target_cost

        penalty_arg = lam + rho * (qc_risk - h)
        penalty = 0.5 / rho * (torch.clamp(penalty_arg, min=0.0) ** 2 - lam ** 2)

        LA = -Q_min + penalty
        return LA, Q_min, qc_risk
    
    def compute_energy(self, state, action):
        return self.compute_LA(state, action)
    
    def compute_score_target_mc(self, state, a_tau, tau):
        """
        For each sample (s, a^τ, τ), we sample N actions a^{0,(i)} around a^τ and compute: 
            φ*(s, a^τ, τ) ≈ - (1/β) * Σ_i w_i ∇_a L_A(s, a^{0,(i)}, λ)
        """
        
        B, act_dim = a_tau.shape
        N = self.score_mc_samples
        device = state.device

        state_exp = state.unsqueeze(1).expand(B, N, -1).reshape(B * N, -1)

        a_tau_exp = a_tau.unsqueeze(1).expand(B, N, -1)

        sigma_all = self.policy.ve_sigmas
        sigma_tau = 1.0 * sigma_all[tau]
        sigma_tau = sigma_tau.view(B, 1, 1)

        noise = torch.randn(B, N, act_dim, device=device)
        a0_samples = a_tau_exp + sigma_tau * noise
        a0_samples_flat = a0_samples.reshape(B * N, act_dim)
        a0_samples_flat.requires_grad_(True)

        LA_flat, _, _ = self.compute_energy(state_exp, a0_samples_flat)
        grad_a_flat = torch.autograd.grad(
            outputs=LA_flat.sum(),
            inputs=a0_samples_flat,
            create_graph=False,
            retain_graph=False
        )[0]

        LA = LA_flat.view(B, N)
        grad_a = grad_a_flat.view(B, N, act_dim)

        beta = self.score_beta
        with torch.no_grad():
            weights = torch.softmax(-LA / beta, dim=1)
        weights = weights.unsqueeze(-1)

        phi_star = - (weights * grad_a).sum(dim=1) / beta

        phi_star = phi_star.detach()

        return phi_star


    def train(self, training=True):
        self.training = training
        self.policy.train(training)
        self.critic.train(training)
        self.safety_critics.train(training)

    def select_action(self, state, eval=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if not eval:
            action = self.policy.sample(state)
        else:
            action = self.policy.sample_deterministic(state)
        return action.detach().cpu().numpy()[0]

    def update_critic(self, state, action, reward, cost, next_state, mask):
        next_action = self.policy.sample(next_state)

        # reward critic
        current_Q1, current_Q2 = self.critic(state, action)
        with torch.no_grad():
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_V = torch.min(target_Q1, target_Q2)
        target_Q = reward + (mask * self.discount * target_V)
        target_Q = target_Q.detach()

        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # safety critics
        qc_idxs = np.arange(self.args.qc_ens_size)
        current_QCs = self.safety_critics(state, action)
        with torch.no_grad():
            next_QCs = self.safety_critic_targets(next_state, next_action)
        next_QC_random_max = next_QCs[qc_idxs].max(dim=0, keepdim=True).values

        if self.args.safetygym:
            mask = torch.ones_like(mask).to(self.device)
        next_QC = next_QCs
        target_QCs = cost[None, :, :].repeat(self.args.qc_ens_size, 1, 1) + \
                     (mask[None, :, :].repeat(self.args.qc_ens_size, 1, 1) * self.safety_discount * next_QC)
        safety_critic_loss = F.mse_loss(current_QCs, target_QCs.detach())

        self.safety_critic_optimizer.zero_grad()
        safety_critic_loss.backward()
        self.safety_critic_optimizer.step()

    def update_actor(self, state, action_taken):
        action, a_tau, tau = self.policy.sample_with_traj(state)
        LA, actor_Q, actor_QC = self.compute_energy(state, action)
        actor_loss = LA.mean()

        phi_star = self.compute_score_target_mc(state, a_tau, tau)
        phi_theta = self.policy.score(state, a_tau, tau=tau)

        score_loss = F.mse_loss(phi_theta, phi_star)

        total_loss = 1.0 * actor_loss + 0.1 * score_loss

        self.actor_optimizer.zero_grad()
        total_loss.backward()
        self.actor_optimizer.step()

        with torch.no_grad():
            current_QCs = self.safety_critics(state, action_taken)
            current_std, current_mean = torch.std_mean(current_QCs, dim=0)
            if self.args.qc_ens_size == 1:
                current_std = torch.zeros_like(current_mean).to(self.device)
            current_QC = current_mean + 1.0 * current_std

        self.log_lam_optimizer.zero_grad()
        lam_loss = torch.mean(self.lam * (self.target_cost - current_QC).detach())
        lam_loss.backward()
        self.log_lam_optimizer.step()
        
        with torch.no_grad():
            self.last_log = {
                "lambda": float(self.lam.item()),
                "log_lambda": float(self.log_lam.item()),
                "lambda_loss": float(lam_loss.item()),
                "qc_risk_mean": float(current_QC.mean().item()),
                "violation_mean": float(torch.clamp(current_QC - self.target_cost, min=0.0).mean().item()),
            }


    def update_parameters(self, memory, updates):
        self.update_counter += 1
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        cost_batch = torch.FloatTensor(reward_batch[:, 1]).to(self.device).unsqueeze(1)
        reward_batch = torch.FloatTensor(reward_batch[:, 0]).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)

        self.update_critic(state_batch, action_batch, reward_batch, cost_batch, next_state_batch, mask_batch)
        self.update_actor(state_batch, action_batch)

        if updates % self.critic_target_update_frequency == 0:
            soft_update(self.critic_target, self.critic, self.critic_tau)
            soft_update(self.safety_critic_targets, self.safety_critics, self.critic_tau)

    # Save model
    def save_model(self, save_dir, suffix=""):

        actor_path = save_dir / f"actor_{suffix}.pth"
        critics_path = save_dir / f"critics_{suffix}.pth"
        safetycritics_path = save_dir / f"safetycritics_{suffix}.pth"

        print(f"[Model] Saving models to:\n  {actor_path}\n  {critics_path}\n  {safetycritics_path}")

        torch.save(self.policy.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critics_path)
        torch.save(self.safety_critics.state_dict(), safetycritics_path)


    # Load model
    def load_model(self, actor_path, critics_path, safetycritics_path):
        print('Loading models from {}, {}, and {}'.format(actor_path, critics_path, safetycritics_path))
        if actor_path is not None:
            self.policy.load_state_dict(torch.load(actor_path))
        if critics_path is not None:
            self.critic.load_state_dict(torch.load(critics_path))
        if safetycritics_path is not None:
            self.safety_critics.load_state_dict(torch.load(safetycritics_path))