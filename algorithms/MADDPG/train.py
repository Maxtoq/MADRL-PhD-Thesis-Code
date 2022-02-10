import argparse
import torch
import sys
import os
import numpy as np
from tqdm import tqdm
from gym.spaces import Box
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from utils.buffer import ReplayBuffer
from utils.make_env import get_paths, load_scenario_config, make_parallel_env
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

    env = make_parallel_env(config.env_path, config.n_rollout_threads, 
                            config.seed, config.discrete_action, sce_conf)

    maddpg = MADDPG.init_from_env(
        env, 
        adversary_alg=config.adversary_alg,
        gamma=config.gamma,
        tau=config.tau,
        lr=config.lr,
        hidden_dim=config.hidden_dim,
        shared_params=config.shared_params
    )

    replay_buffer = ReplayBuffer(
        config.buffer_length, 
        maddpg.nagents,
        [obsp.shape[0] for obsp in env.observation_space],
        [acsp.shape[0] if isinstance(acsp, Box) else acsp.n
        for acsp in env.action_space]
    )

    # Compute number of episodes per update
    if config.n_updates is not None:
        eps_per_update = int(config.n_episodes / config.n_updates)
    else:
        eps_per_update = config.eps_per_update
    
    print(f"Starting training for {config.n_episodes} episodes")
    print(f"                  with {config.n_rollout_threads} threads")
    print(f"                  updates every {eps_per_update} episodes")
    print(f"                  with seed {config.seed}")
    target_update_interval = 0
    for ep_i in tqdm(range(0, config.n_episodes, config.n_rollout_threads)):
        #print("Episodes %i-%i of %i" % (ep_i + 1,
        #                                ep_i + 1 + config.n_rollout_threads,
        #                                config.n_episodes))
        obs = env.reset()
        # obs.shape = (n_rollout_threads, nagent)(nobs), nobs differs per agent so not tensor
        maddpg.prep_rollouts(device='cpu')

        explr_pct_remaining = max(0, config.n_exploration_eps - ep_i) / config.n_exploration_eps
        maddpg.scale_noise(config.final_noise_scale + (config.init_noise_scale - config.final_noise_scale) * explr_pct_remaining)
        maddpg.reset_noise()

        # Rollout episode
        for et_i in range(config.episode_length):
            # rearrange observations to be per agent, and convert to torch Variable
            torch_obs = [Variable(torch.Tensor(np.vstack(obs[:, i])),
                                  requires_grad=False)
                         for i in range(maddpg.nagents)]
            # get actions as torch Variables
            torch_agent_actions = maddpg.step(torch_obs, explore=True)
            # convert actions to numpy arrays
            agent_actions = [ac.data.numpy() for ac in torch_agent_actions]
            # rearrange actions to be per environment
            actions = [[ac[i] for ac in agent_actions] for i in range(config.n_rollout_threads)]
            next_obs, rewards, dones, infos = env.step(actions)
            replay_buffer.push(obs, agent_actions, rewards, next_obs, dones)
            if dones.sum(1).all():
                break
            obs = next_obs

        # Training
        if (len(replay_buffer) >= config.batch_size and
            (ep_i % eps_per_update) < config.n_rollout_threads):
            maddpg.prep_training(device=training_device)
            for _ in range(config.n_training_per_updates):
                for a_i in range(maddpg.nagents):
                    sample = replay_buffer.sample(config.batch_size,
                                                    cuda_device=training_device)
                    maddpg.update(sample, a_i, logger=logger)
            target_update_interval += 1
            if (target_update_interval == config.hard_update_interval):
                maddpg.update_all_targets()
                target_update_interval = 0
            maddpg.prep_rollouts(device='cpu')

        # Store reward
        ep_rews = replay_buffer.get_average_rewards(
            config.episode_length * config.n_rollout_threads
            ) 
        for a_i, a_ep_rew in enumerate(ep_rews):
            logger.add_scalar('agent%i/mean_episode_rewards' % a_i, 
                            a_ep_rew / config.n_rollout_threads, ep_i)
        # Save ep number
        with open(str(log_dir / 'ep_nb.txt'), 'w') as f:
            f.write(str(ep_i))

        if ep_i % config.save_interval < config.n_rollout_threads:
            os.makedirs(run_dir / 'incremental', exist_ok=True)
            maddpg.save(run_dir / 'incremental' / ('model_ep%i.pt' % (ep_i + 1)))
            maddpg.save(model_cp_path)

    maddpg.save(model_cp_path)
    env.close()
    logger.export_scalars_to_json(str(log_dir / 'summary.json'))
    logger.close()
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
    parser.add_argument("--n_episodes", default=25000, type=int)
    parser.add_argument("--buffer_length", default=int(1e6), type=int)
    parser.add_argument("--eps_per_update", default=1000, type=int)
    parser.add_argument("--n_updates", default=None, type=int)
    parser.add_argument("--batch_size", default=1024, type=int,
                        help="Batch size for model training")
    parser.add_argument("--n_training_per_updates", default=1, type=int)
    parser.add_argument("--hard_update_interval", type=int, default=2,
                        help="After how many updates the target should be updated")
    parser.add_argument("--n_exploration_eps", default=25000, type=int)
    parser.add_argument("--init_noise_scale", default=0.3, type=float)
    parser.add_argument("--final_noise_scale", default=0.0, type=float)
    parser.add_argument("--save_interval", default=1000, type=int)
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