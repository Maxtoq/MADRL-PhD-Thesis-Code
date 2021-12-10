import argparse
import torch
import sys
import os
import numpy as np
from tqdm import tqdm
from gym.spaces import Box
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from utils.make_env import get_paths, load_scenario_config, make_parallel_env
from utils.config import get_config
from algo.qmix import QMix
from algo.QMixPolicy import QMixPolicy

from offpolicy.utils.util import get_cent_act_dim, get_dim_from_space
from offpolicy.utils.rec_buffer import RecReplayBuffer

USE_CUDA = torch.cuda.is_available()


def run(args):
    parsed_args = get_config(args)

    # Get paths for saving logs and model
    run_dir, model_cp_path, log_dir = get_paths(parsed_args)
    print("Saving model in dir", run_dir)

    # Init summary writer
    logger = SummaryWriter(str(log_dir))

    # Load scenario config
    sce_conf = load_scenario_config(parsed_args, run_dir)

    # Set seeds
    torch.manual_seed(parsed_args.seed)
    np.random.seed(parsed_args.seed)

    # Set cuda and threads
    if parsed_args.cuda and torch.cuda.is_available():
        print("Using GPU...")
        device = torch.device("cuda:0")
        torch.set_num_threads(parsed_args.n_training_threads)
        if parsed_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("Using CPU...")
        device = torch.device("cpu")
        torch.set_num_threads(parsed_args.n_training_threads)

    env = make_parallel_env(parsed_args.env_path, 
                    parsed_args.n_rollout_threads, 
                    parsed_args.seed, parsed_args.discrete_action, 
                    sce_conf)
    num_agents = len(env.observation_space)

    # create policies and mapping fn
    if parsed_args.share_policy:
        policy_info = {
            'policy_0': {
                "cent_obs_dim": get_dim_from_space(
                                    env.share_observation_space[0]),
                "cent_act_dim": get_cent_act_dim(env.action_space),
                "obs_space": env.observation_space[0],
                "share_obs_space": env.share_observation_space[0],
                "act_space": env.action_space[0]}
        }

        def policy_mapping_fn(id): return 'policy_0'
    else:
        print("Training with different policies for each agents isn't supported.")
        exit(0)
        policy_info = {
            'policy_' + str(agent_id): {
                "cent_obs_dim": get_dim_from_space(
                                    env.share_observation_space[agent_id]),
                "cent_act_dim": get_cent_act_dim(env.action_space),
                "obs_space": env.observation_space[agent_id],
                "share_obs_space": env.share_observation_space[agent_id],
                "act_space": env.action_space[agent_id]}
            for agent_id in range(num_agents)
        }

        def policy_mapping_fn(agent_id): return 'policy_' + str(agent_id)
    policy_ids = sorted(list(policy_info.keys()))
    
    config = {
        "args": parsed_args,
        "policy_info": policy_info,
        "policy_mapping_fn": policy_mapping_fn,
        "env": env,
        "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir
    }

    # Policy for each agent
    policies = {p_id: QMixPolicy(config, policy_info[p_id]) 
                    for p_id in policy_ids}
    # Agents ids for each policy
    policy_agents = {p_id: sorted([agent_id for agent_id in range(num_agents)
                                    if policy_mapping_fn(agent_id) == p_id]) 
                        for p_id in policy_ids}

    # Trainer
    trainer = QMix(parsed_args, num_agents, policies, policy_mapping_fn,
                   device=device, episode_length=parsed_args.episode_length)
    
    # Replay buffer
    num_train_episodes = (parsed_args.n_episodes //
                          parsed_args.train_interval_eps)
    buffer = RecReplayBuffer(policy_info,
                            policy_agents,
                            parsed_args.buffer_size,
                            parsed_args.episode_length,
                            False, False,
                            parsed_args.use_reward_normalization)
    
    last_train_ep = 0
    last_hard_update_ep = 0
    for ep_i in tqdm(range(0, parsed_args.n_episodes, 
                        parsed_args.n_rollout_threads)):
        # TRAINING DONE WHEN SHARING POLICY,
        # [TODO] training with different policies accross agents
        p_id = "policy_0"
        policy = policies[p_id]
        # Perform episode
        obs = env.reset()
        # obs.shape = (n_rollout_threads, nagent)(nobs), 
        # nobs differs per agent so not tensor
        trainer.prep_rollout()

        rnn_states_batch = np.zeros(
            (parsed_args.n_rollout_threads * num_agents,
             parsed_args.hidden_size), 
            dtype=np.float32)
        last_acts_batch = np.zeros(
            (parsed_args.n_rollout_threads * num_agents, policy.output_dim), 
            dtype=np.float32)
        episode_obs = {
            p_id : np.zeros((parsed_args.episode_length + 1, 
                             parsed_args.n_rollout_threads, 
                             num_agents, 
                             policy.obs_dim), 
                            dtype=np.float32) 
            for p_id in policy_ids}
        episode_share_obs = {p_id : np.zeros((parsed_args.episode_length + 1, parsed_args.n_rollout_threads, num_agents, policy.central_obs_dim), dtype=np.float32) for p_id in policy_ids}
        episode_acts = {p_id : np.zeros((parsed_args.episode_length, parsed_args.n_rollout_threads, num_agents, policy.output_dim), dtype=np.float32) for p_id in policy_ids}
        episode_rewards = {p_id : np.zeros((parsed_args.episode_length, parsed_args.n_rollout_threads, num_agents, 1), dtype=np.float32) for p_id in policy_ids}
        episode_dones = {p_id : np.ones((parsed_args.episode_length, parsed_args.n_rollout_threads, num_agents, 1), dtype=np.float32) for p_id in policy_ids}
        episode_dones_env = {p_id : np.ones((parsed_args.episode_length, parsed_args.n_rollout_threads, 1), dtype=np.float32) for p_id in policy_ids}
        episode_avail_acts = {p_id : None for p_id in policy_ids}

        for step_i in range(parsed_args.episode_length):
            share_obs = obs.reshape(parsed_args.n_rollout_threads, -1)
            # Copy over agent dim
            share_obs = np.repeat(share_obs[:, np.newaxis, :], 
                                  num_agents, axis=1)
            # group observations from parallel envs into one batch to process
            # at once
            obs_batch = np.concatenate(obs)
            # get actions for all agents to step the env with exploration noise
            acts_batch, rnn_states_batch, _ = policy.get_actions(obs_batch,
                                                last_acts_batch,
                                                rnn_states_batch,
                                                t_env=ep_i,
                                                explore=True)
            acts_batch = acts_batch if isinstance(acts_batch, np.ndarray) else\
                            acts_batch.cpu().detach().numpy()
            # update rnn hidden state
            rnn_states_batch = rnn_states_batch \
                if isinstance(rnn_states_batch, np.ndarray) \
                else rnn_states_batch.cpu().detach().numpy()
            last_acts_batch = acts_batch

            env_acts = np.split(acts_batch, parsed_args.n_rollout_threads)
            # env step and store the relevant episode information
            next_obs, rewards, dones, infos = env.step(env_acts)

            dones_env = np.all(dones, axis=1)
            terminate_episodes = np.any(dones_env) or \
                                 step_i == parsed_args.episode_length - 1

            episode_obs[p_id][step_i] = obs
            episode_share_obs[p_id][step_i] = share_obs
            episode_acts[p_id][step_i] = np.stack(env_acts)
            episode_rewards[p_id][step_i] = rewards
            episode_dones[p_id][step_i] = dones
            episode_dones_env[p_id][step_i] = dones_env

            obs = next_obs

            if terminate_episodes:
                break
        
        episode_obs[p_id][step_i + 1] = obs
        share_obs = obs.reshape(parsed_args.n_rollout_threads, -1)
        share_obs = np.repeat(share_obs[:, np.newaxis, :], num_agents, axis=1)
        episode_share_obs[p_id][step_i + 1] = share_obs

        # push all episodes collected in this rollout step to the buffer
        buffer.insert(parsed_args.n_rollout_threads,
                        episode_obs,
                        episode_share_obs,
                        episode_acts,
                        episode_rewards,
                        episode_dones,
                        episode_dones_env,
                        episode_avail_acts)

        # Save average rewards on this round of training
        average_episode_rewards = np.mean(np.sum(episode_rewards[p_id], 
                                                 axis=0))
        print(ep_i, average_episode_rewards)

        # Training
        if ep_i * parsed_args.episode_length >= parsed_args.batch_size and \
            ep_i - last_train_ep >= parsed_args.train_interval_eps:
            trainer.prep_training()

            for p_id in policy_ids:
                # Sample batch of experience from replay buffer
                sample = buffer.sample(parsed_args.batch_size)

                # Train on batch
                train_info, _, _ = trainer.train_policy_on_batch(sample)

            # Soft update parameters
            if parsed_args.use_soft_update:
                trainer.soft_target_updates()
            # Hard update parameters
            else:
                if (ep_i - last_hard_update_ep >= 
                    parsed_args.hard_update_interval_episode):
                    trainer.hard_target_updates()
                    last_hard_update_ep = ep_i
            last_train_ep = ep_i

        # Log
        logger.add_scalar('agent0/mean_episode_rewards' %  
                            average_episode_rewards, ep_i)
        # # Save ep number
        # with open(str(log_dir / 'ep_nb.txt'), 'w') as f:
        #     f.write(str(ep_i))

        # Save model
        if ep_i % parsed_args.save_interval < parsed_args.n_rollout_threads:
            os.makedirs(run_dir / 'incremental', exist_ok=True)
            for p_id in policy_ids:
                policy_Q = policies[p_id].q_network
                p_save_path = run_dir / 'incremental' / \
                    f'model_ep{ep_i}_{str(p_id)}'
                torch.save(policy_Q.state_dict(), p_save_path)

    torch.save(policy_Q.state_dict(), model_cp_path)
    env.close()
    logger.export_scalars_to_json(str(log_dir / 'summary.json'))
    logger.close()
    print("Model saved in dir", run_dir)


if __name__ == '__main__':
    run(sys.argv[1:])