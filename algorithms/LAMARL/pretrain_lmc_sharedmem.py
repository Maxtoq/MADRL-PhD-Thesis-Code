import gym
import time
import torch
import random
import numpy as np

from tqdm import trange

from src.utils.config import get_config
from src.utils.eval import perform_eval
from src.utils.utils import set_seeds, set_cuda_device
from src.log.train_log import Logger
from src.log.util import get_paths, write_params
from src.log.progress_bar import Progress
from src.envs.make_env import make_env
from src.lmc.lmc_context import LMC


def run():
     # Load config
    parser = get_config()
    cfg = parser.parse_args()

    # Get paths for saving logs and model
    run_dir, model_cp_path, log_dir = get_paths(cfg)
    print("Saving model in dir", run_dir)
    write_params(run_dir, cfg)

    # Init logger
    logger = Logger(cfg, log_dir)

    set_seeds(cfg.seed)

    # Set training device
    device = set_cuda_device(cfg)
    
    # Create train environment
    if not cfg.magym_global_state:
        cfg.magym_global_state = True
    envs, parser = make_env(cfg, cfg.n_parallel_envs)

    if cfg.do_eval:
        eval_envs, eval_parser = make_env(cfg, cfg.n_parallel_envs)

    # Create model
    n_agents = envs.n_agents
    obs_space = envs.observation_space
    shared_obs_space = envs.shared_observation_space
    act_space = envs.action_space
    global_state_dim = envs.global_state_dim
    model = LMC(
        cfg, 
        n_agents, 
        obs_space, 
        shared_obs_space, 
        act_space,
        parser.vocab, 
        global_state_dim, 
        device)

    # Start training
    print(f"Starting training for {cfg.n_steps} frames")
    print(f"                  updates every {cfg.n_parallel_envs} episodes")
    print(f"                  with seed {cfg.seed}")
    # Reset env
    last_save_step = 0
    last_eval_step = 0
    obs, states = envs.reset()
    lang_contexts = model.reset_context()
    n_steps_per_update = cfg.n_parallel_envs * cfg.episode_length
    for s_i in trange(0, cfg.n_steps, n_steps_per_update, ncols=0):
        model.prep_rollout()
        model.reset_policy_buffers()
        for ep_s_i in range(cfg.episode_length):
            # Parse obs
            parsed_obs = parser.get_perfect_messages(obs)
            # Store language inputs in buffer
            model.store_language_inputs(obs, parsed_obs)
            # Perform step
            # Get action
            actions, broadcasts, agent_messages = model.comm_n_act(
                    obs, parsed_obs)
            # Perform action and get reward and next obs
            obs, next_states, rewards, dones, infos = envs.step(actions)

            # Reward communication
            model.eval_comm(rewards, agent_messages, states, dones)
            states = next_states

            env_dones = dones.all(axis=1)
            if True in env_dones:
                lang_contexts = model.reset_context(env_dones)

            # Save data for logging
            logger.count_returns(s_i, rewards, dones)

            # Insert data into policy buffer
            model.store_exp(rewards, dones)

        # Training
        train_losses = model.train(s_i + n_steps_per_update)

        # Log train data
        logger.log_losses(train_losses, s_i + n_steps_per_update)
    
        # # Eval
        # if cfg.do_eval and s_i + n_steps_per_update - last_eval_step > \
        #         cfg.eval_interval:
        #     last_eval_step = s_i + n_steps_per_update
        #     mean_return, success_rate, mean_ep_len = perform_eval(
        #         cfg, model, eval_envs, eval_parser)
        #     logger.log_eval(s_i, mean_return, success_rate, mean_ep_len)
        #     logger.save()

        # Save
        if s_i + n_steps_per_update - last_save_step > cfg.save_interval:
            last_save_step = s_i + n_steps_per_update
            model.save(run_dir / "incremental" / f"model_ep{last_save_step}.pt")
            
    envs.close()
    # Save model and training data
    model.save(run_dir / "model_ep.pt")
    logger.save_n_close()

if __name__ == '__main__':
    run()