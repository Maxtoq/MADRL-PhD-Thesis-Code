import argparse
import os
import json
import torch
import numpy as np
import pandas as pd

from tensorboardX import SummaryWriter
from tqdm import tqdm

from utils.buffer import ReplayBuffer
from utils.make_env import get_paths, load_scenario_config, make_env
from utils.eval import perform_eval_scenar




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_path", type=str, help="Path to the environment",
                    default="coop_push_scenario/coop_push_scenario_sparse.py")
    parser.add_argument("--model_name", type=str, default="TEST",
                        help="Name of directory to store model/training contents")
    parser.add_argument("--seed", default=1, type=int, help="Random seed")
    # Environment
    parser.add_argument("--episode_length", default=100, type=int)
    parser.add_argument("--discrete_action", action='store_true')
    parser.add_argument("--sce_conf_path", type=str, 
                        default="configs/2a_1o_fo_rel.json",
                        help="Path to the scenario config file")
    # Training
    parser.add_argument("--n_frames", default=25000, type=int,
                        help="Number of training frames to perform")
    parser.add_argument("--buffer_length", default=int(1e6), type=int)
    parser.add_argument("--frames_per_update", default=100, type=int)
    parser.add_argument("--batch_size", default=512, type=int,
                        help="Batch size for model training")
    parser.add_argument("--n_explo_frames", default=None, type=int,
                        help="Number of frames where agents explore, if None then equal to n_frames")
    parser.add_argument("--explo_strat", default="sample", type=str)
    parser.add_argument("--init_explo_rate", default=1.0, type=float)
    parser.add_argument("--final_explo_rate", default=0.0, type=float)
    parser.add_argument("--save_interval", default=100000, type=int)
    # Evalutation
    parser.add_argument("--eval_every", type=int, default=None)
    parser.add_argument("--eval_scenar_file", type=str, default=None)
    # Policy hyperparameters
    parser.add_argument("--policy_algo", default="maddpg", type=str)
    parser.add_argument("--hidden_dim", default=64, type=int)
    parser.add_argument("--lr", default=0.0007, type=float)
    parser.add_argument("--tau", default=0.01, type=float)
    parser.add_argument("--gamma", default=0.99, type=float)
    parser.add_argument("--shared_params", action='store_true')
    # Language learning hyperparameters
    parser.add_argument("--temp")
    # LNovelD
    parser.add_argument("--embed_dim", default=16, type=int)
    parser.add_argument("--int_reward_coeff", default=0.1, type=float)
    parser.add_argument("--noveld_lr", default=1e-4, type=float)
    parser.add_argument("--noveld_scale", default=0.5, type=float)
    parser.add_argument("--noveld_trade_off", default=1.0, type=float)
    # Cuda
    parser.add_argument("--cuda_device", default=None, type=str)

    config = parser.parse_args()

    run(config)