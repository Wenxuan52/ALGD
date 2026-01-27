from ast import Raise
import time
import gym
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pandas as pd
import os

import socket
from pathlib import Path
import setproctitle

from agents.replay_memory import ReplayMemory
from sampler.mujoco_env_sampler import MuJoCoEnvSampler
from sampler.safetygym_env_sampler import SafetygymEnvSampler

# Agents
from agents.algd.ALGD_DDPM import ALGDAgent as DDPM_ALGDAgent
from agents.algd.ALGD_VESDE import ALGDAgent as VESDE_ALGDAgent


def train(args, env_sampler, agent, pool, writer=None):
    total_step = 0
    exploration_before_start(args, env_sampler, pool, agent)
    epoch = 0

    history = []

    for _ in range(args.num_epoch):
        sta = time.time()
        epo_len = args.epoch_length
        train_policy_steps = 0
        for i in range(epo_len):
            cur_state, action, next_state, reward, done, info = env_sampler.sample(agent, i)
            pool.push(cur_state, action, reward, next_state, done)

            # train the policy
            if len(pool) > args.min_pool_size:
                train_policy_steps += train_policy_repeats(args, total_step, train_policy_steps, pool, agent)
            total_step += 1

            def evaluate(num_eval_epo=1):
                env_sampler.current_state = None
                avg_return, avg_cost_return = 0, 0
                eval_step = 0
                for _ in range(num_eval_epo):
                    sum_reward, sum_cost = 0, 0
                    eval_step = 0
                    done = False
                    while not done and eval_step < epo_len:
                        _, _, _, reward, done, _ = env_sampler.sample(agent, eval_step, eval_t=True)
                        sum_reward += reward[0]
                        sum_cost += reward[1] if args.safetygym else args.gamma**eval_step * reward[1]
                        eval_step += 1
                    avg_return += sum_reward
                    avg_cost_return += sum_cost
                avg_return /= num_eval_epo
                avg_cost_return /= num_eval_epo
                return avg_return, avg_cost_return

            if total_step % epo_len == 0 or total_step == 1:
                test_reward, test_cost = evaluate(args.num_eval_epochs)
                print('env: {}, exp: {}, step: {}, test_return: {}, test_cost: {}, budget: {}, seed: {}, cuda_num: {}, time: {}s'.format(args.env_name, args.experiment_name, total_step, np.around(test_reward, 2), np.around(test_cost, 2), args.cost_lim, args.seed, args.cuda_num, int(time.time() - sta)))
                if args.use_tensorboard and writer is not None:
                    writer.add_scalar('Eval/return', test_reward, total_step)
                    writer.add_scalar('Eval/cost', test_cost, total_step)
                    if hasattr(agent, "get_last_log"):
                        last = agent.get_last_log()
                        if "lambda" in last:
                            writer.add_scalar("Train/lambda", last["lambda"], total_step)
                        if "violation_mean" in last:
                            writer.add_scalar("Train/violation_mean", last["violation_mean"], total_step)
                
                if args.save_history:
                    row = {
                        "step": total_step,
                        "return": float(test_reward),
                        "cost": float(test_cost),
                    }
                    
                    # Extra information on lambda
                    # 1) 如果 agent 提供 last_log（推荐做法），就把里面的字段并进来
                    if hasattr(agent, "get_last_log"):
                        row.update(agent.get_last_log())

                    # 2) 兜底：至少把 lambda 记下来（如果 agent 有 lam 属性）
                    if hasattr(agent, "lam"):
                        try:
                            row["lambda"] = float(agent.lam.item())
                        except Exception:
                            row["lambda"] = float(agent.lam)

                    history.append(row)
            
        epoch += 1

    if args.save_history and len(history) > 0:
        import datetime
        date_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
        save_dir = Path.cwd() / "results" / args.env_name / args.experiment_name / f"{date_str}_seed{args.seed}"
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / "history.csv"
        pd.DataFrame(history).to_csv(save_path, index=False)
        print(f"[History] Saved training history to {save_path}")
    
    # save network parameters after training
    if args.save_parameters:
        agent.save_model(save_dir)


def exploration_before_start(args, env_sampler, pool, agent):
    for i in range(args.init_exploration_steps):
        cur_state, action, next_state, reward, done, info = env_sampler.sample(agent, i)
        pool.push(cur_state, action, reward, next_state, done)


def train_policy_repeats(args, total_step, train_step, pool, agent):
    if total_step % args.train_every_n_steps > 0:
        return 0

    if train_step > args.max_train_repeat_per_step * total_step:
        return 0

    for i in range(args.num_train_repeat):
        batch_state, batch_action, batch_reward, batch_next_state, batch_done = pool.sample(args.policy_train_batch_size)
        batch_reward, batch_done = np.squeeze(batch_reward), np.squeeze(batch_done)
        batch_done = (~batch_done).astype(int)
        agent.update_parameters((batch_state, batch_action, batch_reward, batch_next_state, batch_done), i)
    return args.num_train_repeat


def main(args):
    torch.set_num_threads(args.n_training_threads)
    run_dir = Path.cwd() / "results" / args.env_name / args.experiment_name

    env = gym.make(args.env_name)
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.safetygym:
        env.seed(args.seed)
    elif not args.safetygym and hasattr(env, 'reset'):
        env.reset(seed=args.seed)
        if hasattr(env.action_space, 'seed'):
            env.action_space.seed(args.seed)
    else:
        Raise("Unknown env type")

    s_dim = env.observation_space.shape[0]
    
    if args.env_name == 'Ant-v3':
        s_dim = int(27)
    elif args.env_name == 'Humanoid-v3':
        s_dim = int(45)

    if not run_dir.exists():
        os.makedirs(str(run_dir))

    if args.use_tensorboard:
        import datetime
        time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
        run_name = f"{time_str}_seed{args.seed}"

        log_dir = (
            Path("/root/tf-logs")
            / args.env_name
            / args.experiment_name
            / run_name
        )
        log_dir.mkdir(parents=True, exist_ok=True)

        writer = SummaryWriter(log_dir=str(log_dir))

    setproctitle.setproctitle(str(args.env_name) + "-" + str(args.seed))

    # Initialize agent based on args.agent
    if args.agent.lower() == 'algd_ddpm':
        agent = DDPM_ALGDAgent(s_dim, env.action_space, args)
    elif args.agent.lower() == 'algd_vesde':
        agent = VESDE_ALGDAgent(s_dim, env.action_space, args)
    else:
        raise ValueError(f"Unknown agent type: {args.agent}")

    # Initial pool for env
    pool = ReplayMemory(args.replay_size)

    # Sampler of environment
    if args.safetygym:
        env_sampler = SafetygymEnvSampler(args, env)
    else:
        env_sampler = MuJoCoEnvSampler(args, env)
    
    if args.use_tensorboard:
        train(args, env_sampler, agent, pool, writer=writer)
    else:
        train(args, env_sampler, agent, pool, writer=None)

    if args.use_tensorboard:
        writer.close()


if __name__ == '__main__':
    from env.constraints import get_threshold
    import safety_gym
    
    import argparse
    base_parser = argparse.ArgumentParser()
    base_parser.add_argument("--agent", type=str, required=True)
    base_parser.add_argument("--env_name", type=str, required=True)
    
    base_args, _ = base_parser.parse_known_args()

    if base_args.agent.lower() == "algd_ddpm":
        from configs.algd_ddpm import ALGD_DDPM_Parser as readParser
    elif base_args.agent.lower() == "algd_vesde":
        from configs.algd_vesde import ALGD_VESDE_Parser as readParser
    else:
        raise ValueError(f"Unknown agent type: {base_args.agent}")
    
    args = readParser()
    if 'Safe' in args.env_name:
        args.constraint_type = 'safetygym'
        args.safetygym = True
        args.epoch_length = 400
    
    args.cost_lim = get_threshold(args.env_name, constraint=args.constraint_type)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_num
    args.seed = torch.randint(0, 10000, (1,)).item()
    main(args)