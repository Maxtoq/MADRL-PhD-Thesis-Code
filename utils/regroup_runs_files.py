import argparse
import os

from shutil import copyfile


"""
Script for regrouping all training data .csv files from run directories
to a single runs_data directory.
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=str,
                        help="Path to directory containing all runs")
    args = parser.parse_args()

    # Make directory to put all runs
    main_dir = os.path.join(args.model_dir, "all_runs_data")
    if not os.path.exists(main_dir):
        os.makedirs(main_dir)

    # For each run directory
    for run_dir in os.listdir(args.model_dir):
        if not str(run_dir).startswith('run'):
            continue
        # Move the data .csv file to main directory
        run_data_file = os.path.join(
            args.model_dir, run_dir, "mean_episode_rewards.csv")
        new_file_path = os.path.join(main_dir, run_dir + ".csv")
        if os.path.exists(run_data_file):
            copyfile(run_data_file, new_file_path)
        else:
            print(run_data_file, "doesn't exits.")