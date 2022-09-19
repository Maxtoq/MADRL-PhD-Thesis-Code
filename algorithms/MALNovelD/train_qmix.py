import argparse
import os
import json
import torch
import numpy as np
import pandas as pd

from tensorboardX import SummaryWriter
from tqdm import tqdm

from model.modules.qmix import QMIX
from utils.buffer import ReplayBuffer
from utils.make_env import get_paths, load_scenario_config, make_env
from utils.eval import perform_eval_scenar
from utils.decay import EpsilonDecay

def run(cfg):
    # Get paths for saving logs and model
    run_dir, model_cp_path, log_dir = get_paths(config)
    print("Saving model in dir", run_dir)

    # Save args in txt file
    with open(os.path.join(run_dir, 'args.txt'), 'w') as f:
        f.write(str(vars(cfg)))

    # Init summary writer
    logger = SummaryWriter(str(log_dir))

    # Load scenario config
    sce_conf = load_scenario_config(config, run_dir)

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # Set training device
    if torch.cuda.is_available():
        if config.cuda_device is None:
            device = 'cuda'
        else:
            device = torch.device(config.cuda_device)
    else:
        device = 'cpu'

    # Create environment
    env = make_env(cfg.env_path, sce_conf, discrete_action=True)

    # Create model
    n_agents = sce_conf["nb_agents"]
    input_dim = env.observation_space[0].shape[0]
    act_dim = env.action_space[0].n
    qmix = QMIX(
        n_agents, input_dim, act_dim, cfg.lr, cfg.gamma, cfg.tau, 
        cfg.hidden_dim, cfg.shared_params, cfg.init_explo_rate)

    # Create replay buffer


    # Get number of exploration steps
    if cfg.n_explo_frames is None:
        cfg.n_explo_frames = cfg.n_frames
    # Set epsilon decay function
    eps_decay = EpsilonDecay(
        cfg.init_explo_rate, cfg.final_explo_rate, 
        cfg.n_explo_frames, cfg.epsilon_decay_fn)

    # Set-up evaluation scenario
    if cfg.eval_every is not None:
        if cfg.eval_scenar_file is None:
            print("ERROR: Evaluation scenario file must be provided with --eval_scenar_file argument")
            exit()
        else:
            # Load evaluation scenario
            with open(cfg.eval_scenar_file, 'r') as f:
                eval_scenar = json.load(f)
            eval_data_dict = {
                "Step": [],
                "Mean return": [],
                "Success rate": [],
                "Mean episode length": []
            }

    # Start training
    print(f"Starting training for {cfg.n_frames} frames")
    print(f"                  updates every {cfg.frames_per_update} frames")
    print(f"                  with seed {cfg.seed}")
    train_data_dict = {
        "Step": [],
        "Episode extrinsic return": [],
        "Success": [],
        "Episode length": []
    }
    # Reset episode data and environment
    ep_returns = np.zeros(n_agents)
    ep_length = 0
    ep_success = False
    obs = env.reset()
    for step_i in tqdm(range(cfg.n_frames)):
        qmix.set_explo_rate(eps_decay.get_explo_rate(step_i))

        # Get actions
        actions = qmix.get_actions(obs, explore=True)
        print(actions)

        return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_path", type=str, help="Path to the environment",
                    default="algorithms/MALNovelD/scenarios/coop_push_scenario_sparse.py")
    parser.add_argument("--model_name", type=str, default="qmix_TEST",
                        help="Name of directory to store model/training contents")
    parser.add_argument("--seed", default=1, type=int, help="Random seed")
    # Environment
    parser.add_argument("--episode_length", default=100, type=int)
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
    parser.add_argument("--init_explo_rate", default=1.0, type=float)
    parser.add_argument("--final_explo_rate", default=0.0, type=float)
    parser.add_argument("--epsilon_decay_fn", default="linear", type=str)
    parser.add_argument("--save_interval", default=100000, type=int)
    # Evalutation
    parser.add_argument("--eval_every", type=int, default=None)
    parser.add_argument("--eval_scenar_file", type=str, default=None)
    # Model hyperparameters
    parser.add_argument("--hidden_dim", default=64, type=int)
    parser.add_argument("--lr", default=0.0007, type=float)
    parser.add_argument("--tau", default=0.01, type=float)
    parser.add_argument("--gamma", default=0.99, type=float)
    parser.add_argument("--shared_params", action='store_true')
    # Cuda
    parser.add_argument("--cuda_device", default=None, type=str)

    config = parser.parse_args()

    run(config)