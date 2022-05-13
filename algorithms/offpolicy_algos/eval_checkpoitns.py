import argparse
import torch
import json
import os
import pandas as pd

from algo.qmix.QMixPolicy import QMixPolicy
from utils.make_env import make_env

from offpolicy.utils.util import get_dim_from_space


def run(args):
    output_file = "eval_perfs.csv"
    eval_perfs_df = pd.DataFrame(columns=["Step", "Mean Return", "Success rate", "Mean Ep Length", "Name"])

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
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Model directory
    parser.add_argument("model_dir", type=str,
                        help="Path to directory containing model checkpoints \
                              and scenario config")
    # Scenario
    parser.add_argument("--env_path", default="coop_push_scenario/coop_push_scenario_sparse",
                        help="Path to the environment")
    # Evaluation scenario
    parser.add_argument('--init_pos_file', default=None,
                        help='JSON file containing initial positions of \
                            entities')
    # Environment
    parser.add_argument("--episode_length", default=100, type=int)
    parser.add_argument("--discrete_action", action='store_false') 
    # Render
    parser.add_argument("--render", action='store_true')
    parser.add_argument("--step_time", default=0.0, type=float)
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