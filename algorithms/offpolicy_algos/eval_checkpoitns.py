import argparse
import torch
import json
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

from algo.qmix.QMixPolicy import QMixPolicy
from utils.make_env import make_env

from offpolicy.utils.util import get_dim_from_space


def run(args):
    output_file_path = os.path.join(args.model_dir, "eval_perfs.csv")
    eval_perfs_df = pd.DataFrame(columns=["Training Eps", "Mean Return", "Success rate", "Mean Ep Length", "Name"])

    # Load scenario config
    sce_conf_path = os.path.join(args.model_dir, "run1", "sce_config.json")
    sce_conf = {}
    if sce_conf_path is not None:
        with open(sce_conf_path) as cf:
            sce_conf = json.load(cf)
            print('Special config for scenario:', args.env_path)
            print(sce_conf)

    # Load evaluation scenario
    with open(args.init_pos_file, 'r') as f:
        init_pos_scenars = json.load(f)

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

    # Get list of run directories
    run_dirs = os.listdir(args.model_dir)
    # Clean
    for r in run_dirs:
        if len(r) > 5:
            run_dirs.remove(r)
    
    # Evaluate each run
    for run in run_dirs:
        run_dir = os.path.join(args.model_dir, run)
        print("\nEvaluating run", run_dir)

        checkpoints_paths = []
        cp_list = os.listdir(os.path.join(run_dir, "incremental"))
        cp_list.sort()
        for cp in cp_list:
            checkpoints_paths.append(os.path.join(run_dir, "incremental", cp))
        checkpoints_paths.append(os.path.join(run_dir, "model.pt"))
        
        # Evaluate each checkpoint
        for cp_i, cp in enumerate(checkpoints_paths):
            # Load parameters in model
            qmix_policy.load_state(cp)
            qmix_policy.q_network.eval()

            # Execute all episodes
            tot_return = 0.0
            n_success = 0.0
            tot_ep_length = 0.0
            for ep_i in range(len(init_pos_scenars)):
                rnn_states_batch = np.zeros(
                    (sce_conf["nb_agents"], qmix_policy.hidden_size), 
                    dtype=np.float32)
                last_acts_batch = np.zeros(
                    (sce_conf["nb_agents"], qmix_policy.output_dim), 
                    dtype=np.float32)
                # Reset environment with initial positions
                obs = env.reset(init_pos=init_pos_scenars[ep_i])
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
                    tot_return += rewards[0]

                    if dones[0]:
                        n_success += 1
                        break
                    obs = next_obs
                tot_ep_length += step_i + 1

            # Save evaluation performance
            eval_perfs = {
                "Training Eps": cp_i * 10000,
                "Mean Return": tot_return / len(init_pos_scenars), 
                "Success rate": n_success / len(init_pos_scenars),
                "Mean Ep Length": tot_ep_length / len(init_pos_scenars), 
                "Name": args.model_dir.split('/')[-1]
            }
            eval_perfs_df = eval_perfs_df.append(eval_perfs, ignore_index=True)
    # Save perfs in csv
    print("Saving evaluation performance to", output_file_path)
    eval_perfs_df.to_csv(output_file_path, index=False)
        
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Model directory
    parser.add_argument("model_dir", type=str,
                        help="Path to directory containing model checkpoints \
                              and scenario config")
    # Scenario
    parser.add_argument("--env_path", default="coop_push_scenario/coop_push_scenario_sparse.py",
                        help="Path to the environment")
    # Evaluation scenario
    parser.add_argument('--init_pos_file', default=None,
                        help='JSON file containing initial positions of \
                            entities')
    # Environment
    parser.add_argument("--episode_length", default=100, type=int)
    parser.add_argument("--discrete_action", action='store_false') 
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

    args = parser.parse_args()

    run(args)