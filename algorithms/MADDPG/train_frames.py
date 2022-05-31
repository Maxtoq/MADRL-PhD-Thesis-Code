import argparse
from matplotlib.pyplot import step
import torch
import sys
import os
import numpy as np
import pandas as pd
from gym.spaces import Box
from tensorboardX import SummaryWriter

from utils.buffer import ReplayBuffer
from utils.rollouts import train_episode
from utils.make_env import get_paths, load_scenario_config, make_parallel_env
from utils.utils import ProgressBar
from maddpg import MADDPG

USE_CUDA = torch.cuda.is_available()


def run(config):
    # Get paths for saving logs and model
    run_dir, model_cp_path, log_dir = get_paths(config)
    print("Saving model in dir", run_dir)

    # Save args in txt file
    with open(os.path.join(run_dir, 'args.txt'), 'w') as f:
        f.write(str(sys.argv))

    # Init summary writer
    logger = SummaryWriter(str(log_dir))

    # Load scenario config
    sce_conf = load_scenario_config(config, run_dir)

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # Set training device
    if USE_CUDA:
        if config.cuda_device is None:
            training_device = 'cuda'
        else:
            training_device = torch.device(config.cuda_device)
    else:
        training_device = 'cpu'
        torch.set_num_threads(config.n_training_threads)

    env = make_parallel_env(config.env_path, 1, config.seed, 
                            config.discrete_action, sce_conf)

    maddpg = MADDPG.init_from_env(
        env, 
        adversary_alg=config.adversary_alg,
        gamma=config.gamma,
        tau=config.tau,
        lr=config.lr,
        hidden_dim=config.hidden_dim,
        shared_params=config.shared_params,
        init_exploration=config.init_exploration
    )

    replay_buffer = ReplayBuffer(
        config.buffer_length, 
        maddpg.nagents,
        [obsp.shape[0] for obsp in env.observation_space],
        [acsp.shape[0] if isinstance(acsp, Box) else acsp.n
        for acsp in env.action_space]
    )

    # Number of explorations frames
    if config.n_exploration_frames is None:
        config.n_exploration_frames = config.n_frames

    # Initialise progress bar
    pb = ProgressBar(config.n_frames)
    
    print(f"Starting training for {config.n_frames} frames")
    print(f"                  updates every {config.frames_per_update} frames")
    print(f"                  with seed {config.seed}")
    train_data_dict = {
        "Step": [],
        "Episode return": [],
        "Success": [],
        "Episode length": []
    }
    step_i = 0
    last_train_i = 0
    last_save_i = 0
    while step_i < config.n_frames:
        # Print progess
        pb.print_progress(step_i)

        # Preparate model for rollout
        maddpg.prep_rollouts(device='cpu')

        # Set exploration noise
        explr_pct_remaining = max(0, config.n_exploration_frames - step_i) / \
            config.n_exploration_frames
        maddpg.scale_noise(config.final_exploration + \
            (config.init_exploration - config.final_exploration) * \
            explr_pct_remaining)
        maddpg.reset_noise()

        # Rollout episode
        ep_returns, ep_length, ep_success = train_episode(
            env, maddpg, replay_buffer, config.episode_length)
        step_i += ep_length
        
        # Training
        if (len(replay_buffer) >= config.batch_size and
           (step_i - last_train_i) >= config.frames_per_update):
            last_train_i = step_i
            maddpg.prep_training(device=training_device)
            for _ in range(config.n_training_per_updates):
                for a_i in range(maddpg.nagents):
                    sample = replay_buffer.sample(config.batch_size,
                                                  cuda_device=training_device)
                    maddpg.update(sample, a_i, logger=logger)
            maddpg.update_all_targets()

        # Log
        # Log in list
        train_data_dict["Step"].append(step_i)
        train_data_dict["Episode return"].append(np.mean(ep_returns))
        train_data_dict["Success"].append(int(ep_success))
        train_data_dict["Episode length"].append(ep_length)
        # Tensorboard
        logger.add_scalar(
            'agent0/episode_return', 
            train_data_dict["Episode return"][-1], 
            train_data_dict["Step"][-1])

        # Save model
        if (step_i - last_save_i) >= config.save_interval:
            last_save_i = step_i
            os.makedirs(run_dir / 'incremental', exist_ok=True)
            maddpg.save(run_dir / 'incremental' / ('model_ep%i.pt' % (step_i)))
            maddpg.save(model_cp_path)

    maddpg.save(model_cp_path)
    env.close()
    # Log Tensorboard
    logger.export_scalars_to_json(str(log_dir / 'summary.json'))
    logger.close()
    # Log csv
    rewards_df = pd.DataFrame(train_data_dict)
    rewards_df.to_csv(str(run_dir / 'mean_episode_rewards.csv'))
    print("Model saved in dir", run_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("env_path", help="Path to the environment")
    parser.add_argument("model_name",
                        help="Name of directory to store " +
                             "model/training contents")
    parser.add_argument("--seed", default=1, type=int, help="Random seed")
    # Environment
    parser.add_argument("--episode_length", default=100, type=int)
    parser.add_argument("--discrete_action", action='store_true')
    parser.add_argument("--sce_conf_path", default=None, type=str,
                        help="Path to the scenario config file")
    # Training
    parser.add_argument("--n_rollout_threads", default=1, type=int)
    parser.add_argument("--n_training_threads", default=6, type=int)
    parser.add_argument("--n_frames", default=25000, type=int)
    parser.add_argument("--buffer_length", default=int(1e6), type=int)
    parser.add_argument("--frames_per_update", default=100, type=int)
    parser.add_argument("--batch_size", default=512, type=int,
                        help="Batch size for model training")
    parser.add_argument("--n_training_per_updates", default=1, type=int)
    parser.add_argument("--n_exploration_frames", default=None, type=int)
    parser.add_argument("--init_exploration", default=0.6, type=float)
    parser.add_argument("--final_exploration", default=0.0, type=float)
    parser.add_argument("--save_interval", default=100000, type=int)
    # Model hyperparameters
    parser.add_argument("--hidden_dim", default=64, type=int)
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--tau", default=0.01, type=float)
    parser.add_argument("--gamma", default=0.95, type=float)
    parser.add_argument("--adversary_alg", default="MADDPG", type=str,
                        choices=['MADDPG', 'DDPG'])
    parser.add_argument("--shared_params", action='store_true')
    # Cuda
    parser.add_argument("--cuda_device", default=None, type=str)

    config = parser.parse_args()

    run(config)