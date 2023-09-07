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
from src.lmc.lmc import LMC




def pretrain_language(env, parser, actor, lang_learner, n_steps=10000, print_progress=True):
    if print_progress:
        progress = Progress(n_steps)
    
    clip_losses, capt_losses, mean_sims = [], [], []
    step_i = 0
    obs = env.reset()
    while step_i < n_steps:
        if print_progress:
            progress.print_progress(step_i)
        # Parse obs
        sent = parser.parse_observations(obs)
        # Store in buffer
        lang_learner.store(obs, sent)
        # Sample actions
        actions = actor.get_actions()
        # Env step
        obs, rewards, dones, infos = env.step(actions)
        # End of episode
        if all(dones):
            clip_loss, capt_loss, mean_sim = lang_learner.train()
            if np.isnan(clip_loss) or np.isnan(capt_loss) or np.isnan(mean_sim):
                print("nan")
                return 1
            clip_losses.append(clip_loss)
            capt_losses.append(capt_loss)
            mean_sims.append(mean_sim)
            obs = env.reset()
        step_i += 1
    env.close()
    return clip_losses, capt_losses, mean_sims


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
    envs, parser = make_env(cfg, cfg.n_rollout_threads)
    
    n_agents = envs.n_agents
    obs_space = envs.observation_space
    shared_obs_space = envs.shared_observation_space
    act_space = envs.action_space
    write_params(run_dir, cfg)

    if cfg.do_eval:
        eval_envs = make_env(cfg, cfg.n_eval_threads)

    # Create model
    model = LMC(cfg, n_agents, obs_space, shared_obs_space, act_space, 
                parser.vocab, device)

    # Start training
    print(f"Starting training for {cfg.n_steps} frames")
    print(f"                  updates every {cfg.n_rollout_threads} episodes")
    print(f"                  with seed {cfg.seed}")
    # Reset env
    last_save_step = 0
    last_eval_step = 0
    obs = envs.reset()
    model.prep_rollout()
    model.start_episode(obs)
    n_steps_per_update = cfg.n_rollout_threads * cfg.episode_length
    for u_i in trange(0, cfg.n_steps, n_steps_per_update, ncols=10):
        for s_i in range(cfg.episode_length):
            # Parse obs
            parsed_obs = parser.get_perfect_messages(obs)
            # Perform step
            # Get action
            output = model.comm_n_act(obs, parsed_obs)
            actions = output[1]
            messages = output[-1]
            # Perform action and get reward and next obs
            obs, rewards, dones, infos = envs.step(actions)

            # TODO: reset context vector for envs that are done
            print(dones.all(axis=1))

            # Save data for logging
            logger.count_returns(u_i + s_i, rewards, dones)

            # Insert data into replay buffer
            rewards = rewards[..., np.newaxis]
            data = (obs, rewards, dones, infos) + output[:-1]
            model.store_exp(data)

        # Training
        train_losses = model.train()
        # Log train data
        logger.log_losses(train_losses, u_i + s_i + 1)
        model.start_episode(obs)
        model.prep_rollout()
        return


    envs.close()

if __name__ == '__main__':
    run()