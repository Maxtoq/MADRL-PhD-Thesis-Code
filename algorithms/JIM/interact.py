import argparse
import keyboard
import random
import torch
import json
import time
import numpy as np

from utils.make_env import make_env
from utils.actors import KeyboardActor, RandomActor

from models.modules.e2s_noveld import E2S_NovelD


class RewardNormalizator:

    def __init__(self):
        self.reward_sum = 0.0
        self.reward_m2 = 0.0
        self.reward_count = 0.0

    def _update_running_mean(self, rewards):
        """
        Updates the running mean of rewards.
        Inputs:
            rewards (numpy.ndarray): List of rewards to use for the update.
        """
        new_count = len(rewards)
        new_sum = sum(rewards)
        new_mean = new_sum / new_count
        if self.reward_count != 0.0:
            curr_mean = self.reward_sum / self.reward_count
        else:
            curr_mean = 0.0
        new_m2 = np.sum((rewards - new_mean) ** 2) + (
            (self.reward_count * new_count)
            / (self.reward_count + new_count)
            * (new_mean - curr_mean) ** 2
        )

        self.reward_count += new_count
        self.reward_sum += new_sum
        self.reward_m2 += new_m2

    def _get_running_std(self):
        """Returns standard deviation of the running mean of the reward."""
        return np.sqrt(self.reward_m2 / self.reward_count)

    def normalize_rewards(self, rewards):
        """
        Updates the running mean of rewards and returns normalize rewards.
        Inputs:
            rewards (list(float)): List of rewards to use for the update.
        Outputs:
            norm_rewards (list(float)): List of normalized rewards.
        """
        rewards = np.array(rewards)
        self._update_running_mean(rewards)
        std = self._get_running_std()
        if std > 0:
            rewards /= std
        return list(rewards)


def run(args):
    # Load scenario config
    sce_conf = {}
    if args.sce_conf_path is not None:
        with open(args.sce_conf_path) as cf:
            sce_conf = json.load(cf)

    # Create environment
    env = make_env(
        args.env_path,
        sce_conf=sce_conf,
        discrete_action=args.discrete_action)
    nb_agents = env.num_agents

    # Load initial positions if given
    if args.sce_init_pos is not None:
        with open(args.sce_init_pos, 'r') as f:
            init_pos_scenar = json.load(f)
    else:
        init_pos_scenar = None

    if args.actors == "manual" :
        actor = KeyboardActor(nb_agents)
    elif args.actors == "random" :
        actor = RandomActor(nb_agents)
    else:
        print("ERROR : Pick correct actors (random or manual)")
        exit(0)

    obs_dim = env.observation_space[0].shape[0]
    act_dim = env.action_space[0].shape[0]
    intrinsic_reward = E2S_NovelD(nb_agents * obs_dim, 5, 64, 512, 0.5, 0.1)

    # ext_rn = RewardNormalizator()
    # int_rn = RewardNormalizator()

    for ep_i in range(args.n_episodes):
        # Reset the environment
        obs = env.reset(init_pos=init_pos_scenar)
        intrinsic_reward.init_new_episode()
        intrinsic_reward.get_reward(
            torch.Tensor(np.concatenate(obs)).unsqueeze(0))

        env.render()
        time.sleep(args.step_time)

        # if parser is not None:
        #     parser.reset(obj_colors, obj_shapes, land_colors, land_shapes)
        for step_i in range(args.ep_length):
            print("Step", step_i)
            print("Observations:", obs)
            # Get action
            actions = actor.get_action()
            print("Actions:", actions)
            next_obs, rewards, dones, infos = env.step(actions)
            int_reward = intrinsic_reward.get_reward(
                torch.Tensor(np.concatenate(next_obs)).unsqueeze(0))
            int_rewards = [int_reward] * 2

            # norm_ext_rewards = ext_rn.normalize_rewards([rewards[0]])
            # norm_int_rewards = int_rn.normalize_rewards([int_rewards[0]])
            print("Extrinsic rewards:", rewards)#, ", norm:", norm_ext_rewards)
            print("Intrinsic rewards:", int_rewards)#, ", norm:", norm_int_rewards)

            env.render()
            time.sleep(args.step_time)

            if dones[0]:
                break
            obs = next_obs
    
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Scenario
    parser.add_argument("--env_path", default="env/coop_push_scenario_sparse.py",
                        help="Path to the environment")
    parser.add_argument("--sce_conf_path", default=None, 
                        type=str, help="Path to the scenario config file")
    parser.add_argument("--sce_init_pos", default=None, 
                        type=str, help="Path to initial positions config file")
    # Environment
    parser.add_argument("--n_episodes", default=1, type=int)
    parser.add_argument("--ep_length", default=1000, type=int)
    parser.add_argument("--discrete_action", action='store_true')
    # Render
    parser.add_argument("--step_time", default=0.1, type=float)
    # Action
    parser.add_argument("--actors", default="manual", type=str, help="Available actors are 'random' or 'manual'")

    args = parser.parse_args()
    run(args)