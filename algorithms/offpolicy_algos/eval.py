import numpy as np
import argparse
import torch
import time
import json
import sys
import os
from torch.autograd import Variable

from algo.qmix.QMixPolicy import QMixPolicy
from utils.make_env import make_env

from offpolicy.utils.util import get_cent_act_dim, get_dim_from_space

def run(args):
    # Load model
    if args.model_dir is not None:
        model_path = os.path.join(args.model_dir, "model.pt")
        sce_conf_path = os.path.join(args.model_dir, "sce_config.json")
    elif args.model_cp_path is not None and args.sce_conf_path is not None:
        model_path = args.model_cp_path
        sce_conf_path = args.sce_conf_path
    else:
        print("ERROR with model paths: you need to provide the path of either \
               the model directory (--model_dir) or the model checkpoint and \
               the scenario config (--model_cp_path and --sce_conf_path).")
        exit(1)
    if not os.path.exists(model_path):
        sys.exit("Path to the model checkpoint %s does not exist" % 
                    model_path)

    # Load scenario config
    sce_conf = {}
    if sce_conf_path is not None:
        with open(sce_conf_path) as cf:
            sce_conf = json.load(cf)
            print('Special config for scenario:', args.env_path)
            print(sce_conf)

    # Seed env
    seed = args.seed if args.seed is not None else np.random.randint(1e9)
    np.random.seed(seed)
    print("Creating environment with seed", seed)

    # Load initial positions if given
    if args.init_pos_file is not None:
        with open(args.init_pos_file, 'r') as f:
            init_pos_scenars = json.load(f)
        n_episodes = len(init_pos_scenars)
    else:
        n_episodes = args.n_episodes
        init_pos_scenars = [None] * n_episodes

    # Create environment
    env = make_env(args.env_path, discrete_action=args.discrete_action, 
                           sce_conf=sce_conf)

    # Create model
    policy_config = {
            "cent_obs_dim": get_dim_from_space(
                                env.share_observation_space[0]),
            "obs_space": env.observation_space[0],
            "act_space": env.action_space[0]
    }
    config = {
        "args": args,
        "device": torch.device("cpu")
    }
    qmix_policy = QMixPolicy(config, policy_config, train=False)
    qmix_policy.load_state(model_path)
    qmix_policy.q_network.eval()

    for ep_i in range(n_episodes):
        rnn_states_batch = np.zeros(
            (sce_conf["nb_agents"], qmix_policy.hidden_size), 
            dtype=np.float32)
        last_acts_batch = np.zeros(
            (sce_conf["nb_agents"], qmix_policy.output_dim), 
            dtype=np.float32)

        obs = env.reset(init_pos=init_pos_scenars[ep_i])
        rew = 0
        for step_i in range(args.episode_length):
            obs_batch = np.array(obs)
            # get actions for all agents to step the env with exploration noise
            acts_batch, rnn_states_batch, _ = qmix_policy.get_actions(
                obs_batch,
                last_acts_batch,
                rnn_states_batch)
            acts_batch = acts_batch if isinstance(acts_batch, np.ndarray) \
                            else acts_batch.cpu().detach().numpy()
            # update rnn hidden state
            rnn_states_batch = rnn_states_batch \
                if isinstance(rnn_states_batch, np.ndarray) \
                else rnn_states_batch.cpu().detach().numpy()
            last_acts_batch = acts_batch
            
            # Environment step
            next_obs, rewards, dones, infos = env.step(acts_batch)
            if args.verbose == 2:
                print("Obs", next_obs)
            if args.verbose >= 1:
                print("Rewards", rewards)
            rew += rewards[0]

            time.sleep(args.step_time)
            env.render()

            if dones[0]:
                break
            obs = next_obs
        
        print(f'Episode {ep_i + 1} finished after {step_i + 1} steps with \
                return {rew}.')
    print("SEED was", seed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("env_path", help="Path to the environment")
    # Model checkpoint
    parser.add_argument("--model_dir", type=str, default=None,
                        help="Path to directory containing model checkpoint \
                             (model.pt) and scenario config (sce_conf.json)")
    parser.add_argument("--model_cp_path", type=str, default=None,
                        help="Path to the model checkpoint")
    parser.add_argument("--sce_conf_path", default=None, type=str,
                        help="Path to the scenario config file")
    # Environment
    parser.add_argument("--seed",default=None, type=int, help="Random seed")
    parser.add_argument("--n_episodes", default=1, type=int)
    parser.add_argument("--episode_length", default=100, type=int)
    parser.add_argument("--discrete_action", action='store_false') 
    # Render
    parser.add_argument("--step_time", default=0.1, type=float)
    # Print
    parser.add_argument("--verbose", type=int, default=1)
    # recurrent parameters
    parser.add_argument('--prev_act_inp', action='store_true', default=False,
                        help="Whether the actor input takes in previous \
                            actions as part of its input")
    parser.add_argument("--use_rnn_layer", action='store_false',
                        default=True, 
                        help='Whether to use a recurrent policy')
    parser.add_argument("--recurrent_N", type=int, default=1)
    # network parameters
    parser.add_argument('--use_orthogonal', action='store_false', 
                        default=True,
                        help="Whether to use Orthogonal initialization for \
                            weights and\ 0 initialization for biases")
    parser.add_argument("--gain", type=float, default=0.01,
                        help="The gain # of last action layer")
    parser.add_argument('--use_feature_normalization', action='store_false',
                        default=True, 
                        help="Whether to apply layernorm to the inputs")
    parser.add_argument('--use_ReLU', action='store_false',
                        default=True, help="Whether to use ReLU")
    parser.add_argument("--use_conv1d", action='store_true',
                        default=False, help="Whether to use conv1d")
    parser.add_argument("--stacked_frames", type=int, default=1,
                        help="Dimension of hidden layers for actor/critic \
                            networks")
    parser.add_argument('--hidden_size', type=int, default=64,
                help="Dimension of hidden layers for actor/critic networks")
    parser.add_argument('--layer_N', type=int, default=1,
                        help="Number of layers for actor/critic networks")
    # Evaluation scenario
    parser.add_argument('--init_pos_file', default=None,
                        help='JSON file containing initial positions of \
                            entities')

    args = parser.parse_args()

    run(args)