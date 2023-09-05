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
    act_space = act_dim = envs.action_space
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
    progress = Progress(cfg.n_steps)
    # Reset env
    step_i = 0
    ep_step_i = 0
    last_save_step = 0
    last_eval_step = 0
    obs = envs.reset()
    model.prep_rollout()
    model.start_episode(obs)
    while step_i < cfg.n_steps:
        progress.print_progress(step_i)
        # Perform step
        # Get action
        output = model.get_actions(ep_step_i)
        actions = output[-1]
        # Perform action and get reward and next obs
        obs, extr_rewards, dones, infos = envs.step(actions)

        # Get intrinsic reward if needed
        if cfg.ir_algo != "none":
            intr_rewards = algo.get_intrinsic_rewards(obs)
        else:
            intr_rewards = np.zeros((cfg.n_rollout_threads, n_agents))
        rewards = extr_rewards + cfg.ir_coeff * intr_rewards


if __name__ == '__main__':
    run()