import gym
import time
import torch
import random
import numpy as np

from tensorboardX import SummaryWriter
from tqdm import tqdm

from config import get_config
from utils import get_paths, write_params
from envs.make_env import make_env


def set_seeds(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def set_cuda_device(cfg):
    if torch.cuda.is_available():
        if cfg.cuda_device is None:
            device = 'cuda'
        else:
            device = torch.device(cfg.cuda_device)
        if cfg.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        device = 'cpu'
    return device


def run():
    # Load config
    parser = get_config()
    cfg = parser.parse_args()

    # Get paths for saving logs and model
    run_dir, model_cp_path, log_dir = get_paths(cfg)
    print("Saving model in dir", run_dir)

    # Init summary writer
    logger = SummaryWriter(str(log_dir))

    set_seeds(cfg.seed)

    # Set training device
    device = set_cuda_device(cfg)
    
    # Create train environment
    if cfg.env_name == "ma_gym":
        env = make_env(cfg, cfg.n_rollout_threads)
        n_agents = env.n_agents
        obs_dim = env.observation_space[0].shape[0]
        act_dim = act_dim = env.action_space[0].n
        write_params(run_dir, cfg)

    print(n_agents, obs_dim, act_dim)

    # Create model
    if "ppo" in cfg.algorithm_name:
        pass

    env.close()


if __name__ == '__main__':
    run()