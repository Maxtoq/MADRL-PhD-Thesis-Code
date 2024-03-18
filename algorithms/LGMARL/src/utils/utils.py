import os
import json
import torch
import random
import numpy as np


def set_seeds(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def set_cuda_device(cfg):
    if torch.cuda.is_available():
        if cfg.cuda_device is None:
            device = torch.device('cuda')
        else:
            device = torch.device(cfg.cuda_device)
        if cfg.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        device = 'cpu'
    return device

def load_args(cfg):
    args_path = os.path.join(cfg.model_dir, "args.txt")
    if not os.path.isfile(args_path):
        print(f"ERROR: args file {args_path} does not exist.")
        exit()
    else:
        with open(args_path, "r") as f:
            [next(f) for i in range(3)]
            args = json.load(f)
        args.pop("seed")
        args.pop("cuda_device")
        args.pop("model_dir")
        args.pop("n_parallel_envs")
        args.pop("use_render")
        args.pop("render_wait_input")
        args.pop("n_steps")
        args.pop("log_comm")
        # if "no_render" in args:
        #     args.pop("no_render")
        for a in args:
            if not hasattr(cfg, a):
                print(f"WARNING: Argument {a} not found in config.")
            else:
                setattr(cfg, a, args[a])