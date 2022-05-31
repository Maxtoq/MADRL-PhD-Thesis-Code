import argparse
import json
import os
import pandas as pd

from maddpg import MADDPG
from utils.make_env import make_env
from algorithms.MADDPG.utils.rollouts import perform_eval_scenar


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
    model_path = os.path.join(args.model_dir, "run1", "model.pt")
    maddpg = MADDPG.init_from_save(model_path)
    maddpg.prep_rollouts(device='cpu')

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
        for i in range(len(cp_list)):
            if i not in [0, 1, 12, 23, 34, 45, 56, 67, 78, 89, 101]:
                continue
            checkpoints_paths.append(
                os.path.join(run_dir, "incremental", cp_list[i]))
        checkpoints_paths.append(os.path.join(run_dir, "model.pt"))
        
        # Evaluate each checkpoint
        for cp_i, cp in enumerate(checkpoints_paths):
            # Load parameters in model
            maddpg.load_cp(cp)
            maddpg.prep_rollouts(device='cpu')

            # Execute all episodes
            mean_return, success_rate, mean_ep_length = perform_eval_scenar(
                env, maddpg, init_pos_scenars, args.episode_length)

            # Save evaluation performance
            eval_perfs = {
                "Training Eps": cp_i * 10000,
                "Mean Return": mean_return, 
                "Success rate": success_rate,
                "Mean Ep Length": mean_ep_length, 
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
    
    args = parser.parse_args()

    run(args)