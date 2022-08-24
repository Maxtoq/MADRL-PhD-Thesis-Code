import shutil
import sys
import os
import re

"""
Must be called as:
python move_run_data.py *path_to_model_dir* *run_names*
*run_names* being a list of either single runs (e.g. 'run12'), or 
sequences of runs noted as 'runN-runM' which corresponds to all runs 
between N and M.
"""

if __name__ == '__main__':
    # Get args
    model_dir = sys.argv[1]
    runs = sys.argv[2:]

    # Get run paths
    run_dirs = []
    for r in runs:
        # Sequence of runs
        if '-' in r:
            # Get run numbers
            nbs = re.findall("run(\d+)", r)
            run_nbs = list(range(int(nbs[0]), int(nbs[1]) + 1))
            # Create run paths
            for nb in run_nbs:
                run_path = os.path.join(model_dir, "run" + str(nb))
                if os.path.isdir(run_path):
                    run_dirs.append(run_path)
                else:
                    print("WARNING:", run_path, "does not exist.")
        # Single run
        else:
            run_path = os.path.join(model_dir, r)
            if os.path.isdir(run_path):
                run_dirs.append(run_path)
            else:
                print("WARNING:", run_path, "does not exist.")
    
    # Create data dir
    data_dir = os.path.join(model_dir, "to_plot_01")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # Move train data to new directory
    for run in run_dirs:
        file_path = os.path.join(run, "training_data.csv")
        if not os.path.exists(file_path):
            print("WARNING: data file", file_path, "does not exist.")
        run_name = re.findall("(run\d+)", run)[0] + ".csv"
        shutil.copyfile(file_path, os.path.join(data_dir, run_name))