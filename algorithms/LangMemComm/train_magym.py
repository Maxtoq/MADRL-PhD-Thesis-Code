import gym
import time
import torch
import random
import numpy as np

from tensorboardX import SummaryWriter
from tqdm import tqdm
from itertools import chain

from config import get_config
from log import get_paths, write_params
from envs.make_env import make_env
from algorithms.mappo.mappo import MAPPO


def set_seeds(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def set_cuda_device(cfg):
    if torch.cuda.is_available():
        if cfg.cuda_device is None:
            device = torch.device('cuda')
        else:
            device = torch.device(cfg.cuda_device)
        if cfg.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        device = 'cpu'
    return device

def reset_envs(envs):
    obs = envs.reset()
    share_obs = []
    for o in obs:
        share_obs.append(list(chain(*o)))
    share_obs = np.array(share_obs)
    return obs, share_obs


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
        envs = make_env(cfg, cfg.n_rollout_threads)
        n_agents = envs.n_agents
        obs_space = envs.observation_space
        shared_obs_space = envs.shared_observation_space
        act_space = act_dim = envs.action_space
        write_params(run_dir, cfg)

    # Create model
    if "ppo" in cfg.algorithm_name:
        algo = MAPPO(
            cfg, n_agents, obs_space, shared_obs_space, act_space, device)

    if cfg.use_eval:
        eval_data_dict = {
            "Step": [],
            "Mean return": [],
            "Success rate": [],
            "Mean episode length": []
        }

    # Start training
    print(f"Starting training for {cfg.n_steps} frames")
    print(f"                  updates every {cfg.n_steps_per_update} frames")
    print(f"                  with seed {cfg.seed}")
    train_data_dict = {
        "Step": [],
        "Episode return": [],
        "Success": [],
        "Episode length": []
    }
    # Reset env
    obs, share_obs = reset_envs(envs)
    ep_step_i = 0
    algo.warmup(obs, share_obs)
    for step_i in tqdm(range(0, int(cfg.n_steps), cfg.n_rollout_threads)):
        algo.prep_rollout()
        # Perform step
        # Get action
        output = algo.get_actions(ep_step_i)
        actions = output[-1]
        # Perform action and get reward and next obs
        # print(actions)
        obs, rewards, dones, infos = envs.step(actions)
        # Insert data into replay buffer
        data = (obs, rewards, dones, infos) + output[:-1]
        algo.store(data)

        # Check for end of episode
        done = False
        print(dones)
        return
        if all(dones) or ep_step_i + 1 == cfg.episode_length:
            done = True

        # If end of episode
        if done:
            algo.train()
        else:
            ep_step_i += 1
    env.close()


if __name__ == '__main__':
    run()