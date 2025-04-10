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

# def get_shape_from_obs_space(obs_space):
#     if obs_space.__class__.__name__ == 'Box':
#         obs_shape = obs_space.shape
#     elif obs_space.__class__.__name__ == 'list':
#         obs_shape = obs_space
#     else:
#         raise NotImplementedError
#     return obs_shape

# def get_shape_from_act_space(act_space):
#     if act_space.__class__.__name__ == 'Discrete':
#         act_shape = 1
#     elif act_space.__class__.__name__ == "MultiDiscrete":
#         act_shape = act_space.shape
#     elif act_space.__class__.__name__ == "Box":
#         act_shape = act_space.shape[0]
#     elif act_space.__class__.__name__ == "MultiBinary":
#         act_shape = act_space.shape[0]
#     else:  # agar
#         act_shape = act_space[0].shape[0] + 1  
#     return act_shape

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
        args.pop("render_wait_input")
        if "no_render" in args:
            args.pop("no_render")
        for a in args:
            assert hasattr(cfg, a), f"Argument {a} not found in config."
            setattr(cfg, a, args[a])