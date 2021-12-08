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

    # Trainer
    trainer = QMix(parsed_args, num_agents, policies, policy_mapping_fn,
                   device=device, episode_length=parsed_args.episode_length)
    
    # Replay buffer
    num_train_episodes = (parsed_args.n_episodes //
                          parsed_args.train_interval_eps)
    print(parsed_args.n_episodes, parsed_args.train_interval_eps, num_train_episodes)
    exit(0)

    maddpg = MADDPG.init_from_env(
        env, 
        agent_alg=config.agent_alg,
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
    
    t = 0
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
            if dones[0,0]:
                break
            obs = next_obs
            t += config.n_rollout_threads
            if (len(replay_buffer) >= config.batch_size and
                (t % config.steps_per_update) < config.n_rollout_threads):
                if USE_CUDA:
                    maddpg.prep_training(device='gpu')
                else:
                    maddpg.prep_training(device='cpu')
                for u_i in range(config.n_rollout_threads):
                    for a_i in range(maddpg.nagents):
                        sample = replay_buffer.sample(config.batch_size,
                                                      to_gpu=USE_CUDA)
                        maddpg.update(sample, a_i, logger=logger)
                    maddpg.update_all_targets()
                maddpg.prep_rollouts(device='cpu')
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
    run(sys.argv[1:])