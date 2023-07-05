import argparse
import shutil
import sys
import os
import re

"""
Must be called as:
$ python move_run_data.py *path_to_model_dir* --data_file *data_fil_name* --runs *run_names*
*run_names* being a list of either single runs (e.g. 'run12'), or 
sequences of runs noted as 'runN-runM' which corresponds to all runs 
between N and M.
"""

if __name__ == '__main__':
    # Get args
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=str)
    parser.add_argument("--data_file", type=str, default="training_data.csv")
    parser.add_argument("--output_dir_name", type=str, default="train_data")
    parser.add_argument("--runs", type=str, nargs="*", required=True)
    args = parser.parse_args()

    # Get run paths
    run_dirs = []
    for r in args.runs:
        # Sequence of runs
        if '-' in r:
            # Get run numbers
            nbs = re.findall("run(\d+)", r)
            run_nbs = list(range(int(nbs[0]), int(nbs[1]) + 1))
            # Create run paths
            for nb in run_nbs:
                run_path = os.path.join(args.model_dir, "run" + str(nb))
                if os.path.isdir(run_path):
                    run_dirs.append(run_path)
                else:
                    print("WARNING:", run_path, "does not exist.")
        # Single run
        else:
            run_path = os.path.join(args.model_dir, r)
            if os.path.isdir(run_path):
                run_dirs.append(run_path)
            else:
                print("WARNING:", run_path, "does not exist.")
    
    # Create data dir
    data_dir = os.path.join(args.model_dir, args.output_dir_name)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # Move train data to new directory
    for run in run_dirs:
        source_file_path = os.path.join(run, "logs", args.data_file)
        if not os.path.exists(source_file_path):
            print("WARNING: data file", source_file_path, "does not exist.")
        run_name = re.findall("(run\d+)", run)[0] + ".csv"
        dest_file_path = os.path.join(data_dir, run_name)
        while os.path.isfile(dest_file_path):
            dest_file_path = dest_file_path[:-4] + "_1.csv"
        shutil.copyfile(source_file_path, os.path.join(data_dir, run_name))