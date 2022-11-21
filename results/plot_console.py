import termplotlib as tpl
import pandas as pd
import numpy as np
import argparse

def mov_avg(data, step_range=10000):
    index_range = int(step_range / 30) + 1
    drop_ids = []
    start_index = data.index[0]
    for s_i in range(0, data["Step"].iloc[-1], step_range):
        rows = data[(data.index >= start_index) & (data.index < start_index + index_range)]["Step"]
        ids = list(rows[(rows  >= s_i) & (rows  < s_i + step_range)].index)
        
        if len(ids) == 0:
            print("hin?")
            continue
        
        keep_id = ids[0]
        drop_ids += ids[1:]
        # For all numerical columns but the 'Step'
        for c in data.select_dtypes(include=np.number).columns:
            if c == "Step":
                continue
            data.at[keep_id, c] = data.loc[ids][c].mean()
        data.at[keep_id, "Step"] = s_i + step_range
        
        start_index = ids[-1] + 1
    return data.drop(drop_ids)

def run(cfg):
    # Load data
    data = pd.read_csv(cfg.run_data_path)

    if cfg.avg_step_range is not None:
        print("Computing moving average of all numerical columns...")
        data = mov_avg(data, step_range=cfg.avg_step_range)
        print("DONE")

    for c in ["Episode extrinsic return", "Episode intrinsic return", "Episode return"]:
        fig = tpl.figure()
        fig.plot(data["Step"].tolist(), data[c].tolist(), title=c, width=100, height=20)
        fig.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_data_path", type=str)
    parser.add_argument("--data_to_plot", type=str, default="Episode extrinsic return")
    parser.add_argument("--avg_step_range", type=int, default=None)
    args = parser.parse_args()
    run(args)
