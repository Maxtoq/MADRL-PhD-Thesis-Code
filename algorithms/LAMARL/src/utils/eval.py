import time
import numpy as np

from ..envs.make_env import make_env

def perform_eval(args, model, envs, parser, render=False):
    returns = np.zeros(args.n_eval_threads)
    success = [False] * args.n_eval_threads
    ep_lengths = np.ones(args.n_eval_threads) * args.episode_length

    obs = envs.reset()
    model.prep_rollout()
    model.start_episode(obs)
    for s_i in range(cfg.episode_length):
            # Parse obs
            parsed_obs = parser.get_perfect_messages(obs)
            # Perform step
            # Get action
            _, actions, _, _, _, messages = model.comm_n_act(obs, parsed_obs)
            # Perform action and get reward and next obs
            obs, rewards, dones, _ = envs.step(actions)

            global_rewards = rewards.mean(axis=1)
            global_dones = dones.all(axis=1)
            for e_i in range(args.n_eval_threads):
                if not success[e_i]:
                    returns[e_i] += global_rewards[e_i]
                    if global_dones[e_i]:
                        success[e_i] = True
                        ep_lengths[e_i] = step_i + 1

            if render and args.n_eval_threads == 1:
                envs.render()
                time.sleep(0.1)

    mean_return = np.mean(returns)
    success_rate = np.mean(success)
    mean_ep_lengths = np.mean(ep_lengths)

    return mean_return, success_rate, mean_ep_lengths
