import argparse

def ALGD_VESDE_Parser():
    parser = argparse.ArgumentParser(description='ALGD_VESDE')
    # ---------------------Agent Config-----------------------
    parser.add_argument('--agent', default='algd_vesde', type=str)

    # ----------------------Env Config------------------------
    parser.add_argument('--env_name', default='Hopper-v3')
    # MuJoCo: 'Hopper-v3' 'HalfCheetah-v3' 'Ant-v3' 'Humanoid-v3'
    # Safety-Gym: 'Safexp-PointButton1-v0' 'Safexp-CarButton1-v0' 'Safexp-PointButton2-v0' 'Safexp-CarButton2-v0' 'Safexp-PointPush1-v0'
    parser.add_argument('--safetygym', action='store_true', default=False)
    parser.add_argument('--constraint_type', default='velocity', help="['safetygym', 'velocity']")
    parser.add_argument('--epoch_length', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=123456)

    # -------------------Experiment Config---------------------
    parser.add_argument('--cuda', default=True, action="store_true", help='run on CUDA (default: True)')
    parser.add_argument('--cuda_num', default='0', help='select the cuda number you want your program to run on')
    parser.add_argument('--use_tensorboard', action='store_true', default=False)
    parser.add_argument('--n_training_threads', default=10)
    parser.add_argument('--experiment_name', default='exp')
    parser.add_argument('--num_epoch', type=int, default=300)
    parser.add_argument('--num_eval_epochs', type=int, default=1)
    parser.add_argument('--save_parameters', action='store_true', default=False)
    parser.add_argument('--save_history', action='store_true', default=False)

    # ---------------------Algorithm Config-------------------------
    parser.add_argument('--qc_ens_size', type=int, default=4)
    parser.add_argument('--score_mc_samples', type=int, default=4)
    parser.add_argument('--rho', type=float, default=1.0)

    parser.add_argument('--diffusion_steps_K', type=int, default=5)
    parser.add_argument('--beta', type=float, default=1.0)

    parser.add_argument('--critic_hidden_size', type=int, default=256)
    parser.add_argument('--score_model_hidden_dim', type=int, default=128)

    # -------------------Training Config---------------------
    parser.add_argument('--init_exploration_steps', type=int, default=5000)
    parser.add_argument('--train_every_n_steps', type=int, default=1)
    parser.add_argument('--num_train_repeat', type=int, default=10)
    parser.add_argument('--critic_target_update_frequency', type=int, default=2)
    parser.add_argument('--replay_size', type=int, default=1000000)
    parser.add_argument('--min_pool_size', type=int, default=1000)
    parser.add_argument('--max_train_repeat_per_step', type=int, default=5)
    
    # -------------------RL Config---------------------
    parser.add_argument('--safety_gamma', type=float, default=0.99)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--tau', type=float, default=0.005)

    parser.add_argument('--lr', type=float, default=0.0003)
    parser.add_argument('--qc_lr', type=float, default=0.0003)
    parser.add_argument('--policy_train_batch_size', type=int, default=256)

    return parser.parse_args()
