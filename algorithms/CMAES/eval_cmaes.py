import argparse
import torch
import json
import sys
import os
import numpy as np
from torch.autograd import Variable
from utils.make_env import make_env
from train_cmaes import PolicyNetwork

def run(config):
    # Load model
    if not os.path.exists(config.model_cp_path):
        sys.exit("Path to the model checkpoint %s does not exist" % 
                    config.model_cp_path)

    # Load scenario config
    sce_conf = {}
    if config.sce_conf_path is not None:
        with open(config.sce_conf_path) as cf:
            sce_conf = json.load(cf)
            print('Special config for scenario:', config.env_path)
            print(sce_conf)

    # Initiate env
    env = make_env(config.env_path, sce_conf, 
                   discrete_action=config.discrete_action)

    # Create model
    num_in_pol = env.observation_space[0].shape[0]
    if config.discrete_action:
        num_out_pol = env.action_space[0].n
    else:
        num_out_pol = env.action_space[0].shape[0]
    policy = PolicyNetwork(num_in_pol, num_out_pol, config.hidden_dim,  
                           discrete_action=config.discrete_action)
    policy.load_state_dict(torch.load(config.model_cp_path))
    policy.eval()

    for ep_i in range(config.n_episodes):
        obs = env.reset()
        episode_reward = 0.0
        for step_i in range(config.episode_length):
            # Rearrange observations to fit in the model
            torch_obs = Variable(torch.Tensor(np.vstack(obs)),
                                    requires_grad=False)
            
            actions = policy(torch_obs)

            # Convert actions to numpy arrays
            agent_actions = [ac.data.numpy() for ac in actions]

            next_obs, rewards, dones, infos = env.step(agent_actions)

            episode_reward += sum(rewards) / sce_conf['nb_agents']

            env.render()

            if dones[0]:
                break

            obs = next_obs
        
        print(f'Episode {ep_i + 1} finished after {step_i + 1} steps with return {episode_reward}.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("env_path", help="Path to the environment")
    parser.add_argument("model_cp_path", type=str,
                        help="Path to the model checkpoint")
    parser.add_argument("--seed",default=1, type=int, help="Random seed")
    parser.add_argument("--n_episodes", default=1, type=int)
    parser.add_argument("--episode_length", default=100, type=int)
    parser.add_argument("--hidden_dim", default=32, type=int)
    parser.add_argument("--sce_conf_path", default=None, type=str,
                        help="Path to the scenario config file")
    parser.add_argument("--discrete_action", action='store_true')

    config = parser.parse_args()

    run(config)