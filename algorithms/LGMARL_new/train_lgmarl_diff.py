import os
import gym
import time
import torch
import random
import numpy as np

from tqdm import trange

from src.utils.config import get_config
from src.utils.utils import set_seeds, set_cuda_device, load_args
from src.utils.decay import ParameterDecay
from src.log.train_log import Logger
from src.log.util import get_paths, write_params
from src.log.progress_bar import Progress
from src.envs.make_env import make_env
from src.algo.lgmarl_diff import LanguageGroundedMARL


def run():
     # Load config
    argparse = get_config()
    cfg = argparse.parse_args()

    # Load pretrained checkpoint if needed
    if cfg.model_dir is not None:
        # Get pretrained stuff
        assert cfg.model_dir is not None, "Must provide model_dir"
        steps_done = load_args(cfg)
        pretrained_model_path = os.path.join(cfg.model_dir, "model_ep.pt")
        assert os.path.isfile(pretrained_model_path), "No model checkpoint provided."
        print("Starting from pretrained model with config:")
        print(cfg)
        comm_eps_start = 1.0
    else:
        comm_eps_start = 1.0

    # Set training step (if continue previous run)
    if cfg.continue_run:
        start_step = steps_done
        if cfg.n_steps <= start_step:
            print("No need to train more.")
            exit()
    else:
        start_step = 0

    # Get paths for saving logs and model
    run_dir, log_dir = get_paths(cfg)
    print("Saving model in dir", run_dir)
    write_params(run_dir, cfg)

    # Init logger
    logger = Logger(cfg, log_dir, start_step)

    set_seeds(cfg.seed)

    # Set training device
    device = set_cuda_device(cfg)
    
    # Create train environment
    envs, parser = make_env(cfg, cfg.n_parallel_envs)

    # Create model
    n_agents = envs.n_agents
    obs_space = envs.observation_space
    shared_obs_space = envs.shared_observation_space
    act_space = envs.action_space
    model = LanguageGroundedMARL(
        cfg, 
        n_agents, 
        obs_space, 
        shared_obs_space, 
        act_space,
        parser, 
        device,
        log_dir,
        comm_eps_start)

    # Load params
    if cfg.model_dir is not None:
        model.load(pretrained_model_path)

    # Start training
    print(f"Starting training for {cfg.n_steps - start_step} frames")
    print(f"                  updates every {cfg.n_parallel_envs} episodes")
    print(f"                  with seed {cfg.seed}")
    # Reset env
    last_save_step = 0
    last_eval_step = 0
    obs = envs.reset()
    parsed_obs = parser.get_perfect_messages(obs)
    model.init_episode(obs, parsed_obs)
    n_steps_per_update = cfg.n_parallel_envs * cfg.rollout_length
    for s_i in trange(start_step, cfg.n_steps, n_steps_per_update, ncols=0):
        model.prep_rollout(device)

        for ep_s_i in range(cfg.rollout_length):
            # Perform step
            # Get action
            actions, broadcasts, agent_messages, comm_rewards \
                = model.act()

            # Perform action and get reward and next obs
            obs, rewards, dones, infos = envs.step(actions)

            env_dones = dones.all(axis=1)
            # if True in env_dones:
            #     model.reset_context(env_dones)

            # Log rewards
            logger.count_returns(s_i, rewards, dones)
            logger.log_comm(
                s_i + ep_s_i * cfg.n_parallel_envs, comm_rewards)

            # Insert data into policy buffer
            parsed_obs = parser.get_perfect_messages(obs)
            model.store_exp(obs, parsed_obs, rewards, dones)

        # Training
        train_losses = model.train(
            s_i + n_steps_per_update,
            train_lang=not cfg.FT_freeze_lang)

        # Log train data
        logger.log_losses(train_losses, s_i + n_steps_per_update)

        # Reset buffer for new episode
        model.init_episode()

        # Save
        if s_i + n_steps_per_update - last_save_step > cfg.save_interval:
            last_save_step = s_i + n_steps_per_update
            if cfg.save_increments:
                model.save(run_dir / "incremental" / f"model_ep{last_save_step}.pt")
            
    envs.close()
    # Save model and training data
    logger.save_n_close()
    model.save(run_dir / "model_ep.pt")

if __name__ == '__main__':
    run()