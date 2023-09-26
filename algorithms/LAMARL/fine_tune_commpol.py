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


def run():
     # Load config
    parser = get_config()
    cfg = parser.parse_args()

    # Check comm_pol param is right
    if cfg.comm_policy_algo in ["no_comm", "perfect_comm"]:
        r = None
        while r not in ["", "n", "N", "y", "Y"]:
            r = input("WARNING: Fine-tuning communication with a non-parametric communication policy? REALLY? (y/N)")
            if r in ["n", "N", ""]:
                exit()
            elif r in ["y", "Y"]:
                print("ok then...")

    pretrained_model_path = cfg.FT_pretrained_model_path
    if pretrained_model_path is None:
        print("Must provide path of pretrained model.")
        exit()

    # Get paths for saving logs and model
    run_dir, model_cp_path, log_dir = get_paths(cfg)
    print("Saving model in dir", run_dir)

    # Init logger
    logger = Logger(cfg, log_dir)

    set_seeds(cfg.seed)

    # Set training device
    device = set_cuda_device(cfg)
    
    # Create train environment
    envs, parser = make_env(cfg, cfg.n_parallel_envs)
    
    n_agents = envs.n_agents
    obs_space = envs.observation_space
    shared_obs_space = envs.shared_observation_space
    act_space = envs.action_space
    write_params(run_dir, cfg)

    if cfg.do_eval:
        eval_envs, eval_parser = make_env(cfg, cfg.n_parallel_envs)

    # Create model
    model = LMC(cfg, n_agents, obs_space, shared_obs_space, act_space, 
                parser.vocab, device)

    # Load params
    model.load(pretrained_model_path)

    # Start training
    print(f"Starting training for {cfg.n_steps} frames")
    print(f"                  updates every {cfg.n_parallel_envs} episodes")
    print(f"                  with seed {cfg.seed}")
    # Reset env
    last_save_step = 0
    last_eval_step = 0
    obs = envs.reset()
    lang_contexts = model.reset_context()
    model.start_episode()
    n_steps_per_update = cfg.n_parallel_envs * cfg.episode_length
    for s_i in trange(0, cfg.n_steps, n_steps_per_update, ncols=0):
        model.prep_rollout()
        for ep_s_i in range(cfg.episode_length):
            # # Parse obs
            # parsed_obs = parser.get_perfect_messages(obs)
            # # Store language inputs in buffer
            # model.store_language_inputs(obs, parsed_obs)
            # Perform step
            # Get action
            values, actions, action_log_probs, rnn_states, rnn_states_critic, \
                messages, lang_contexts = model.comm_n_act(obs, lang_contexts)
            # Perform action and get reward and next obs
            obs, rewards, dones, infos = envs.step(actions)

            # Save data for logging
            logger.count_returns(s_i, rewards, dones)

            env_dones = dones.all(axis=1)
            if True in env_dones:
                break
                lang_contexts = model.reset_context(lang_contexts, env_dones)

            # Reward communication
            mean_message_return = model.eval_comm(rewards)

            # Train comm
            comm_losses = model.train_comm()

            # Log communication reward and loss
            logger.log_comm(
                s_i + ep_s_i * cfg.n_parallel_envs, 
                mean_message_return, comm_losses)

            # Insert data into replay buffer
            rewards = rewards[..., np.newaxis]
            model.store_exp(rewards, dones, infos, values, 
                actions, action_log_probs, rnn_states, rnn_states_critic)

        # Training policy
        if s_i + n_steps_per_update > cfg.FT_n_steps_fix_policy:
            train_losses = model.train(
                s_i + n_steps_per_update, train_lang=False)
            # Log train data
            logger.log_losses(train_losses, s_i + n_steps_per_update)

        # Reset policy buffer
        model.start_episode()
    
        # Eval
        if cfg.do_eval and s_i + n_steps_per_update - last_eval_step > \
                cfg.eval_interval:
            last_eval_step = s_i + n_steps_per_update
            mean_return, success_rate, mean_ep_len = perform_eval(
                cfg, model, eval_envs, eval_parser)
            logger.log_eval(s_i, mean_return, success_rate, mean_ep_len)
            logger.save()

        # Save
        if s_i + n_steps_per_update - last_save_step > cfg.save_interval:
            last_save_step = s_i + n_steps_per_update
            model.save(run_dir / "incremental" / ('model_ep%i.pt' % (s_i)))
            
    envs.close()
    # Save model and training data
    model.save(run_dir / "model_ep.pt")
    logger.save_n_close()

if __name__ == '__main__':
    run()