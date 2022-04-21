import argparse
import torch
import sys
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from gym.spaces import Box
from torch.autograd import Variable
from tensorboardX import SummaryWriter

from utils.make_env import get_paths, load_scenario_config, make_parallel_env
from utils.config import get_config

from offpolicy.utils.util import get_cent_act_dim, get_dim_from_space, DecayThenFlatSchedule


def run(args):
    parsed_args = get_config(args)

    # Get paths for saving logs and model
    run_dir, model_cp_path, log_dir = get_paths(parsed_args)
    print("Saving model in dir", run_dir)

    # Save args in txt file
    with open(os.path.join(run_dir, 'args.txt'), 'w') as f:
        f.write(str(sys.argv))

    # Init summary writer
    logger = SummaryWriter(str(log_dir))

    # Load scenario config
    sce_conf = load_scenario_config(parsed_args, run_dir)

    # Set seeds
    torch.manual_seed(parsed_args.seed)
    np.random.seed(parsed_args.seed)

    # Set cuda and threads
    if parsed_args.cuda and torch.cuda.is_available():
        if parsed_args.cuda_device is None:
            device = torch.device("cuda:0")
        else:
            device = torch.device(parsed_args.cuda_device)
        print("Using GPU, CUDA device", device)
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
        #"policy_info": policy_info,
        #"policy_mapping_fn": policy_mapping_fn,
        #"env": env,
        #"num_agents": num_agents,
        "device": device,
        #"run_dir": run_dir
    }

    # Get wanted trainer and policy
    if parsed_args.algorithm_name == "qmix":
        from algo.qmix.QMixPolicy import QMixPolicy as Policy
        from algo.qmix.qmix import QMix as TrainAlgo
    elif parsed_args.algorithm_name == "maddpg":
        from algo.maddpg.MADDPGPolicy import MADDPGPolicy as Policy
        from algo.maddpg.maddpg import MADDPG as TrainAlgo
    elif parsed_args.algorithm_name == "matd3":
        from algo.maddpg.MADDPGPolicy import MATD3Policy as Policy
        from algo.maddpg.maddpg import MATD3 as TrainAlgo

    # Policy for each agent
    policies = {p_id: Policy(config, policy_info[p_id])
                    for p_id in policy_ids}
    # Agents ids for each policy
    policy_agents = {p_id: sorted([agent_id for agent_id in range(num_agents)
                                    if policy_mapping_fn(agent_id) == p_id])
                        for p_id in policy_ids}

    # Trainer
    trainer = TrainAlgo(
        parsed_args, 
        num_agents, 
        policies, 
        policy_mapping_fn,
        device=device, 
        episode_length=parsed_args.episode_length)

    # Compute number of episodes per update
    if parsed_args.n_updates is not None:
        num_train_episodes = parsed_args.n_updates
        eps_per_update = int(parsed_args.n_episodes / parsed_args.n_updates)
    else:
        num_train_episodes = int(
            parsed_args.n_episodes / parsed_args.train_interval_eps)
        eps_per_update = parsed_args.train_interval_eps

    # Imports for RNN or MLP models
    if parsed_args.algorithm_name in ["qmix"]:
        from buffer.rec_buffer import RecReplayBuffer as ReplayBuffer
        from buffer.rec_buffer import PrioritizedRecReplayBuffer as PrioReplayBuffer
        from runner.rnn_runner import RNNRunner as Runner
    elif parsed_args.algorithm_name in ["maddpg", "matd3"]:
        from buffer.mlp_buffer import MlpReplayBuffer as ReplayBuffer
        from buffer.mlp_buffer import PrioritizedMlpReplayBuffer as PrioReplayBuffer
        from runner.mlp_runner import MLPRunner as Runner
    
    # Replay buffer
    if parsed_args.use_per:
        beta_anneal = DecayThenFlatSchedule(
            parsed_args.per_beta_start, 
            1.0, num_train_episodes, 
            decay="linear")
        buffer = PrioReplayBuffer(
            parsed_args.per_alpha,
            policy_info,
            policy_agents,
            parsed_args.buffer_size,
            parsed_args.episode_length,
            False, False,
            parsed_args.use_reward_normalization)
    else:
        buffer = ReplayBuffer(
            policy_info,
            policy_agents,
            parsed_args.buffer_size,
            parsed_args.episode_length,
            False, False,
            parsed_args.use_reward_normalization)

    # Rollout runner
    runner = Runner(parsed_args, num_agents, policies["policy_0"], env, buffer)
    
    print(f"Starting training for {parsed_args.n_episodes} episodes")
    print(f"                  with {parsed_args.n_rollout_threads} threads")
    print(f"                  updates every {eps_per_update} episodes")
    print(f"                  with seed {parsed_args.seed}")
    train_step = 0
    last_hard_update_ep = 0
    train_data_dict = {
        "Step": [],
        "Episode return": [],
        "Success": [],
        "Episode length": []
    }
    for ep_i in tqdm(range(0, parsed_args.n_episodes, 
                        parsed_args.n_rollout_threads)):
        # TRAINING DONE WHEN SHARING POLICY
        trainer.prep_rollout()

        # Rollout episode
        episode_return, ep_dones, ep_length = runner.train_rollout(ep_i)

        # Training
        if (ep_i * parsed_args.episode_length >= parsed_args.batch_size and
            (ep_i % eps_per_update) < parsed_args.n_rollout_threads):
            trainer.prep_training()

            for p_id in policy_ids:
                # Sample batch of experience from replay buffer
                if parsed_args.use_per:
                    beta = beta_anneal.eval(train_step)
                    sample = buffer.sample(parsed_args.batch_size, beta, p_id)
                else:
                    sample = buffer.sample(parsed_args.batch_size)

                # Train on batch
                train_info, new_priorities, idxes = \
                    trainer.train_policy_on_batch(sample, "policy_0")

                if parsed_args.use_per:
                    buffer.update_priorities(idxes, new_priorities, p_id)

            # Soft update parameters
            if parsed_args.use_soft_update:
                trainer.soft_target_updates()
            # Hard update parameters
            else:
                if (ep_i - last_hard_update_ep >= 
                    parsed_args.hard_update_interval_episode):
                    trainer.hard_target_updates()
                    last_hard_update_ep = ep_i
            train_step += 1

        # Log
        for r_i in range(parsed_args.n_rollout_threads):
            # Log Tensorboard
            logger.add_scalar('agent0/mean_episode_rewards',
                            np.mean(episode_return[r_i]), ep_i + r_i)
            # Log in list
            train_data_dict["Step"].append(ep_i + r_i)
            train_data_dict["Episode return"].append(np.mean(episode_return[r_i]))
            train_data_dict["Success"].append(ep_dones[r_i])
            train_data_dict["Episode length"].append(ep_length[r_i])
        
        # # Save ep number
        # with open(str(log_dir / 'ep_nb.txt'), 'w') as f:
        #     f.write(str(ep_i))

        # Save model
        if ep_i % parsed_args.save_interval < parsed_args.n_rollout_threads:
            os.makedirs(run_dir / 'incremental', exist_ok=True)
            for p_id in policy_ids:
                p_save_path = run_dir / 'incremental' / \
                    f'model_ep{ep_i}_{str(p_id)}'
                policies[p_id].save_state(p_save_path)

    # Save model
    policies[p_id].save_state(model_cp_path)

    env.close()
    logger.export_scalars_to_json(str(log_dir / 'summary.json'))
    logger.close()
    # Log csv
    rewards_df = pd.DataFrame(train_data_dict)
    rewards_df.to_csv(str(run_dir / 'mean_episode_rewards.csv'))
    print("Model saved in dir", run_dir)


if __name__ == '__main__':
    run(sys.argv[1:])