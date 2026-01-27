import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

# Initialize Policy weights for ensemble networks
def init_weights(m):
    def truncated_normal_init(t, mean=0.0, std=0.01):
        torch.nn.init.normal_(t, mean=mean, std=std)
        while True:
            cond = torch.logical_or(t < mean - 2 * std, t > mean + 2 * std)
            if not torch.sum(cond):
                break
            t = torch.where(cond, torch.nn.init.normal_(torch.ones(t.shape), mean=mean, std=std), t)
        return t

    if type(m) == nn.Linear or isinstance(m, EnsembleFC):
        input_dim = m.in_features
        truncated_normal_init(m.weight, std=1 / (2 * np.sqrt(input_dim)))
        m.bias.data.fill_(0.0)


class QNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(QNetwork, self).__init__()

        # Q1 architecture
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        # Q2 architecture
        self.linear4 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state, action):
        xu = torch.cat([state, action], 1)
        
        x1 = F.relu(self.linear1(xu))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)

        x2 = F.relu(self.linear4(xu))
        x2 = F.relu(self.linear5(x2))
        x2 = self.linear6(x2)
        return x1, x2

class EnsembleFC(nn.Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    ensemble_size: int
    weight: torch.Tensor

    def __init__(self, in_features: int, out_features: int, ensemble_size: int, weight_decay: float = 0., bias: bool = True) -> None:
        super(EnsembleFC, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ensemble_size = ensemble_size
        self.weight = nn.Parameter(torch.Tensor(ensemble_size, in_features, out_features))
        self.weight_decay = weight_decay
        if bias:
            self.bias = nn.Parameter(torch.Tensor(ensemble_size, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        pass

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        w_times_x = torch.bmm(input, self.weight)
        return torch.add(w_times_x, self.bias[:, None, :])

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

class QcEnsemble(nn.Module):
    def __init__(self, state_size, action_size, ensemble_size, hidden_size=256):
        super(QcEnsemble, self).__init__()
        self.nn1 = EnsembleFC(state_size + action_size, hidden_size, ensemble_size, weight_decay=0.00003)
        self.nn2 = EnsembleFC(hidden_size, hidden_size, ensemble_size, weight_decay=0.00006)
        self.nn3 = EnsembleFC(hidden_size, 1, ensemble_size, weight_decay=0.0001)
        self.activation = nn.SiLU()
        self.ensemble_size = ensemble_size
        self.apply(init_weights)

    def forward(self, state, action):
        xu = torch.cat([state, action], 1)
        nn1_output = self.activation(self.nn1(xu[None, :, :].repeat([self.ensemble_size, 1, 1])))
        nn2_output = self.activation(self.nn2(nn1_output))
        nn3_output = self.nn3(nn2_output)

        return nn3_output

    def get_decay_loss(self):
        decay_loss = 0.
        for m in self.children():
            if isinstance(m, EnsembleFC):
                decay_loss += m.weight_decay * torch.sum(torch.square(m.weight)) / 2.
        return decay_loss

# -----------------------------------------------------------------------------

class DiffusionPolicy(nn.Module):
    """
    DDPM diffusion policy
    """
    def __init__(self,
                 num_inputs,
                 num_actions,
                 hidden_dim=128,
                 T=10,
                 action_space=None):
        super(DiffusionPolicy, self).__init__()

        self.state_dim = num_inputs
        self.action_dim = num_actions
        self.hidden_dim = hidden_dim
        self.T = T

        # diffusion schedule
        beta_start = 1e-4
        beta_end = 0.02
        betas = torch.linspace(beta_start, beta_end, T)

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.tensor([1.0]), alphas_cumprod[:-1]], dim=0
        )

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod",
            torch.sqrt(1.0 - alphas_cumprod)
        )

        # time embedding
        self.time_embed_dim = 16
        self.time_embedding = nn.Embedding(T, self.time_embed_dim)

        # MLP
        input_dim = self.state_dim + self.action_dim + self.time_embed_dim
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        layers.append(nn.ReLU())
        self.mlp = nn.Sequential(*layers)

        self.eps_head = nn.Linear(hidden_dim, self.action_dim)
        
        self.score_head = nn.Linear(hidden_dim, self.action_dim)

        self.apply(weights_init_)
        
        # guidance
        self.guidance_scale = 0.05
        self.guidance_normalize = True

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.
            )
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.
            )

    def _extract(self, a, t, x_shape):
        out = a.gather(-1, t.long())
        return out.view(-1, *([1] * (len(x_shape) - 1)))

    def q_sample(self, x0, noise, t):
        """
        q(x_t | x_0) = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
        """
        sqrt_alphas_cumprod_t = self._extract(
            self.sqrt_alphas_cumprod, t, x0.shape
        )
        sqrt_one_minus_alphas_cumprod_t = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x0.shape
        )
        return sqrt_alphas_cumprod_t * x0 + \
            sqrt_one_minus_alphas_cumprod_t * noise

    def forward(self, state, x_t, t):
        if t.dim() == 0:
            t = t.unsqueeze(0).repeat(state.size(0))
        t_embed = self.time_embedding(t)
        h = torch.cat([state, x_t, t_embed], dim=-1)
        h = self.mlp(h)
        eps = self.eps_head(h)
        return eps
    
    def score(self, state, action, tau=None):
        """
        score API:
        - Input: (s, a, tau)
        - Return: φ_theta(s,a,tau), for -∇_a L_A(s,a,λ).
        """
        B = state.size(0)

        if tau is None:
            tau = torch.zeros(B, dtype=torch.long, device=state.device)
        else:
            if tau.dim() == 0:
                tau = tau.unsqueeze(0).repeat(B)

        t_embed = self.time_embedding(tau)
        h = torch.cat([state, action, t_embed], dim=-1)
        h = self.mlp(h)
        phi_theta = self.score_head(h)
        return phi_theta

    # Reverse sampling
    def p_sample(self, state, x_t, t):
        b = x_t.size(0)
        t_batch = torch.full((b,), t, dtype=torch.long, device=x_t.device)

        beta_t = self._extract(self.betas, t_batch, x_t.shape)
        alpha_t = self._extract(self.alphas, t_batch, x_t.shape)
        alpha_cumprod_t = self._extract(self.alphas_cumprod, t_batch, x_t.shape)
        alpha_cumprod_prev_t = self._extract(
            self.alphas_cumprod_prev, t_batch, x_t.shape
        )

        eps_theta = self.forward(state, x_t, t_batch)

        # DDPM
        x0_pred = (
            x_t - torch.sqrt(1.0 - alpha_cumprod_t) * eps_theta
        ) / torch.sqrt(alpha_cumprod_t + 1e-8)

        # posterior mean
        coef1 = torch.sqrt(alpha_cumprod_prev_t) * beta_t / (1.0 - alpha_cumprod_t + 1e-8)
        coef2 = torch.sqrt(alpha_t) * (1.0 - alpha_cumprod_prev_t) / (1.0 - alpha_cumprod_t + 1e-8)
        mean = coef1 * x0_pred + coef2 * x_t

        # guidance
        if self.guidance_scale != 0.0:
            a_t = torch.tanh(x_t) * self.action_scale + self.action_bias

            # φ_theta(s, a_t, t)
            phi = self.score(state, a_t, tau=t_batch)

            if self.guidance_normalize:
                phi = phi / (phi.norm(dim=-1, keepdim=True) + 1e-8)

            mean = mean + self.guidance_scale * phi

        if t == 0:
            return mean

        noise = torch.randn_like(x_t)
        var = beta_t * (1.0 - alpha_cumprod_prev_t) / (1.0 - alpha_cumprod_t + 1e-8)
        std = torch.sqrt(var + 1e-8)

        return mean + std * noise

    def p_sample_loop(self, state, steps=None):
        if steps is None:
            steps = self.T

        b = state.size(0)
        x_t = torch.randn(b, self.action_dim, device=state.device)

        for i in reversed(range(steps)):
            x_t = self.p_sample(state, x_t, i)

        # back to action space
        action = torch.tanh(x_t) * self.action_scale + self.action_bias
        return action

    # API
    def sample(self, state, steps=None):
        return self.p_sample_loop(state, steps=steps)
    
    def sample_with_traj(self, state, steps=None):
        """
        Sample action along with an intermediate action in the trajectory.
        - Return: a0, a_tau, tau
            a0: action sampled at t=0
            a_tau: action sampled at t=τ
            tau: the random timestep τ
        """
        if steps is None:
            steps = self.T

        b = state.size(0)
        device = state.device

        x_t = torch.randn(b, self.action_dim, device=device)

        taus = torch.randint(low=0, high=steps, size=(b,), device=device)
        a_tau = torch.zeros(b, self.action_dim, device=device)

        for i in reversed(range(steps)):
            x_t = self.p_sample(state, x_t, i)

            mask = (taus == i)
            if mask.any():
                a_t = torch.tanh(x_t) * self.action_scale + self.action_bias
                a_tau[mask] = a_t[mask]

        a0 = torch.tanh(x_t) * self.action_scale + self.action_bias
        return a0, a_tau, taus


    def sample_deterministic(self, state, steps=None):
        """
        For evaluation
        """
        if steps is None:
            steps = self.T

        b = state.size(0)
        x_t = torch.zeros(b, self.action_dim, device=state.device)

        for i in reversed(range(steps)):
            x_t = self.p_sample(state, x_t, i)

        action = torch.tanh(x_t) * self.action_scale + self.action_bias
        return action

    def get_a_mean(self, state):
        return self.sample_deterministic(state)

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(DiffusionPolicy, self).to(device)
