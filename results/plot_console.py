import termplotlib as tpl
import pandas as pd
import argparse


def run(cfg):
    # Load data
    data = pd.load_csv(cfg.run_data_path)
    print(data.head)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_data_path", type=str)
    parser.add_argument("--x_max", type=int, default=None)
    args = parser.parse_args()
    run(args)
