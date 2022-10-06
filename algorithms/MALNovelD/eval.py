import argparse
import json
import sys
import os
import numpy as np

from utils.make_env import make_env
from utils.eval import eval_episode, rnn_eval_episode


def run(cfg):
     # Get paths
    if config.model_dir is not None:
        model_path = os.path.join(config.model_dir, "model.pt")
        sce_conf_path = os.path.join(config.model_dir, "sce_config.json")
    elif config.model_cp_path is not None and config.sce_conf_path is not None:
        model_path = config.model_cp_path
        sce_conf_path = config.sce_conf_path
    else:
        sys.exit("ERROR with model paths: you need to provide the path of either \
                  the model directory (--model_dir) or the model checkpoint and \
                  the scenario config (--model_cp_path and --sce_conf_path).")
    if not os.path.exists(model_path):
        sys.exit("Path to the model checkpoint %s does not exist" % model_path)

    # Get model class
    if "maddpg" in model_path:
        if "manoveld" in model_path:
            from model.modules.maddpg_noveld import MADDPG_MANovelD as Model
        elif "panoveld" in model_path:
            from model.modules.maddpg_noveld import MADDPG_PANovelD as Model
        elif "mpanoveld" in model_path:
            from model.modules.maddpg_noveld import MADDPG_MPANovelD as Model
        else:
            from model.modules.maddpg import MADDPG as Model
        eval_fn = eval_episode
    elif "qmix" in model_path:
        if "manoveld" in model_path:
            from model.modules.qmix_noveld import QMIX_MANovelD as Model
        else:
            from model.modules.qmix import QMIX as Model
        eval_fn = rnn_eval_episode
    
    # Create model
    model = Model.init_from_save(model_path)
    model.prep_rollouts(device="cpu")

    # Load scenario config
    sce_conf = {}
    if sce_conf_path is not None:
        with open(sce_conf_path) as cf:
            sce_conf = json.load(cf)
            print('Special config for scenario:', config.env_path)
            print(sce_conf)

    # Load initial positions if given
    if config.init_pos_file is not None:
        with open(config.init_pos_file, 'r') as f:
            init_pos_scenars = json.load(f)
        n_episodes = len(init_pos_scenars)
    else:
        n_episodes = config.n_episodes
        init_pos_scenars = [None] * n_episodes

    # Seed env
    seed = config.seed if config.seed is not None else np.random.randint(1e9)
    np.random.seed(seed)
    print("Creating environment with seed", seed)

    # Create environment
    env = make_env(config.env_path, discrete_action=config.discrete_action, 
                    sce_conf=sce_conf)

    for ep_i in range(n_episodes):
        ep_return, ep_length, ep_success = eval_fn(
            env, 
            model, 
            config.episode_length,
            init_pos_scenars[ep_i],
            render=True,
            step_time=config.step_time,
            verbose=True)
        
        print(f'Episode {ep_i + 1} finished after {ep_length} steps with \
                return {ep_return}.')
    print("SEED was", seed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("env_path", help="Path to the environment")
    # Model checkpoint
    parser.add_argument("--model_dir", type=str, default=None,
                        help="Path to directory containing model checkpoint \
                             (model.pt) and scenario config (sce_conf.json)")
    parser.add_argument("--model_cp_path", type=str, default=None,
                        help="Path to the model checkpoint")
    parser.add_argument("--sce_conf_path", default=None, type=str,
                        help="Path to the scenario config file")
    # Environment
    parser.add_argument("--seed",default=None, type=int, help="Random seed")
    parser.add_argument("--n_episodes", default=1, type=int)
    parser.add_argument("--episode_length", default=100, type=int)
    parser.add_argument("--discrete_action", action='store_true')
    # Render
    parser.add_argument("--step_time", default=0.1, type=float)
    # Evaluation scenario
    parser.add_argument('--init_pos_file', default=None,
                        help='JSON file containing initial positions of \
                            entities')

    config = parser.parse_args()

    run(config)