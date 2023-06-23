import time
import numpy as np

from ..envs.make_env import make_env, reset_envs

def perform_eval(args, algo, render=False):
    # Create env
    envs = make_env(args, args.n_eval_threads, args.seed + 1000)

    returns = np.zeros(args.n_eval_threads)
    success = [False] * args.n_eval_threads
    ep_lengths = np.ones(args.n_eval_threads) * args.episode_length

    obs, share_obs = reset_envs(envs)
    algo.start_episode(obs, share_obs)
    algo.prep_rollout()
    for step_i in range(args.episode_length):
        # Get action
        output = algo.get_actions(step_i)
        actions = output[-1]
        # Perform action and get reward and next obs
        obs, rewards, dones, infos = envs.step(actions)

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
