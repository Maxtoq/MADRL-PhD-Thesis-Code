import gym
import time
import torch
import random
import numpy as np

from tqdm import tqdm

from src.utils.config import get_config
from src.utils.eval import perform_eval
from src.log.train_log import Logger
from src.log.util import get_paths, write_params
from src.log.progress_bar import Progress
from src.envs.make_env import make_env
from src.mappo.mappo import MAPPO
from src.mappo.mappo_intrinsic import MAPPO_IR


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


def run():
    # Load config
    parser = get_config()
    cfg = parser.parse_args()

    # Get paths for saving logs and model
    run_dir, model_cp_path, log_dir = get_paths(cfg)
    print("Saving model in dir", run_dir)

    # Init logger
    logger = Logger(cfg, log_dir)

    set_seeds(cfg.seed)

    # Set training device
    device = set_cuda_device(cfg)
    
    # Create train environment
    # if cfg.env_name == "ma_gym":
    envs = make_env(cfg, cfg.n_rollout_threads)
    if cfg.env_name == "rel_overgen":
        cfg.episode_length = cfg.ro_state_dim
    n_agents = envs.n_agents
    obs_space = envs.observation_space
    shared_obs_space = envs.shared_observation_space
    act_space = act_dim = envs.action_space
    write_params(run_dir, cfg)

    if cfg.do_eval:
        eval_envs = make_env(cfg, cfg.n_eval_threads)

    # Create model
    if "ppo" in cfg.algorithm_name:
        if cfg.ir_algo == "none":
            algo = MAPPO(
                cfg, n_agents, obs_space, shared_obs_space, act_space, device)
        else:
            algo = MAPPO_IR(
                cfg, n_agents, obs_space, shared_obs_space, act_space, device)

    # Start training
    print(f"Starting training for {cfg.n_steps} frames")
    print(f"                  updates every {cfg.n_rollout_threads} episodes")
    print(f"                  with seed {cfg.seed}")
    progress = Progress(cfg.n_steps)
    # Reset env
    step_i = 0
    ep_step_i = 0
    last_save_step = 0
    last_eval_step = 0
    obs = envs.reset()
    algo.prep_rollout()
    algo.start_episode(obs, cfg.n_rollout_threads)
    while step_i < cfg.n_steps:
        progress.print_progress(step_i)
        # Perform step
        # Get action
        output = algo.get_actions(ep_step_i)
        actions = output[-1]
        # Perform action and get reward and next obs
        obs, extr_rewards, dones, infos = envs.step(actions)

        # Get intrinsic reward if needed
        if cfg.ir_algo != "none":
            intr_rewards = algo.get_intrinsic_rewards(obs)
        else:
            intr_rewards = np.zeros((cfg.n_rollout_threads, n_agents))
        rewards = extr_rewards + cfg.ir_coeff * intr_rewards

        # Save data for logging
        logger.count_returns(
            ep_step_i, rewards, extr_rewards, intr_rewards, dones)

        # Insert data into replay buffer
        rewards = rewards[..., np.newaxis]
        data = (obs, rewards, dones, infos) + output[:-1]
        algo.store(data)

        # Check for end of episode
        done = False
        if dones.all(axis=1).all() or ep_step_i + 1 == cfg.episode_length:
            done = True

        # If end of episode
        if done:
            train_losses = algo.train()
            # Log train data
            step_i += logger.log_train(step_i, train_losses)
            # Reset env (env, buffer, log)
            obs = envs.reset()
            algo.start_episode(obs, cfg.n_rollout_threads)
            logger.reset_episode()
            algo.prep_rollout()
            ep_step_i = 0
        else:
            ep_step_i += 1

        # Eval
        if cfg.do_eval and step_i - last_eval_step > cfg.eval_interval:
            last_eval_step = step_i
            mean_return, success_rate, mean_ep_len = perform_eval(cfg, algo)
            logger.log_eval(step_i, mean_return, success_rate, mean_ep_len)

        # Save
        if step_i - last_save_step > cfg.save_interval:
            last_save_step = step_i
            algo.save(run_dir / "incremental" / ('model_ep%i.pt' % (step_i)))
            logger.save()
            algo.prep_rollout(device)

    progress.print_end()
    envs.close()
    # Save model and training data
    algo.save(run_dir / "model_ep.pt")
    logger.save_n_close()


if __name__ == '__main__':
    run()
