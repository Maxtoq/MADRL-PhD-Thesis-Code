import argparse
import os
import json
import torch
import numpy as np
import pandas as pd

from tensorboardX import SummaryWriter
from tqdm import tqdm

from model.modules.maddpg_noveld import MADDPG_PANovelD, MADDPG_MANovelD, MADDPG_MPANovelD
from utils.buffer import ReplayBuffer
from utils.make_env import get_paths, load_scenario_config, make_env
from utils.eval import perform_eval_scenar


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
    env = make_env(cfg.env_path, sce_conf, cfg.discrete_action)

    # Create model
    nb_agents = sce_conf["nb_agents"]
    input_dim = env.observation_space[0].shape[0]
    if cfg.discrete_action:
        act_dim = env.action_space[0].n
    else:
        act_dim = env.action_space[0].shape[0]
    if cfg.noveld_type == "multi_agent":
        NoveldClass = MADDPG_MANovelD
    elif cfg.noveld_type == "per_agent":
        NoveldClass = MADDPG_PANovelD
    elif cfg.noveld_type == "multi+per_agent":
        NoveldClass = MADDPG_MPANovelD
    else:
        print("ERROR: bad noveld type.")
    maddpg = NoveldClass(
        nb_agents, input_dim, act_dim, cfg.lr, cfg.gamma, 
        cfg.tau, cfg.hidden_dim, cfg.embed_dim, cfg.discrete_action, 
        cfg.shared_params, cfg.init_explo_rate, cfg.explo_strat)
    maddpg.prep_rollouts(device='cpu')
    
    # Create replay buffer
    replay_buffer = ReplayBuffer(
        cfg.buffer_length, 
        nb_agents,
        [obsp.shape[0] for obsp in env.observation_space],
        [acsp.shape[0] if not cfg.discrete_action else acsp.n
            for acsp in env.action_space]
    )

    # Get number of exploration steps
    if cfg.n_explo_frames is None:
        cfg.n_explo_frames = cfg.n_frames

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
        "Episode return": [],
        "Episode extrinsic return": [],
        "Episode intrinsic return": [],
        "Success": [],
        "Episode length": []
    }
    # Reset episode data and environment
    ep_returns = np.zeros(nb_agents)
    ep_ext_returns = np.zeros(nb_agents)
    ep_int_returns = np.zeros(nb_agents)
    ep_length = 0
    ep_success = False
    obs = env.reset()
    for step_i in tqdm(range(cfg.n_frames)):
        # Compute and set exploration rate
        explo_pct_remaining = \
            max(0, cfg.n_explo_frames - step_i) / cfg.n_explo_frames
        maddpg.scale_noise(cfg.final_explo_rate + 
            (cfg.init_explo_rate - cfg.final_explo_rate) * explo_pct_remaining)

        # Perform step
        obs = np.array(obs)
        torch_obs = torch.Tensor(obs)
        actions = maddpg.step(torch_obs, explore=True)
        actions = [a.squeeze().data.numpy() for a in actions]
        next_obs, ext_rewards, dones, _ = env.step(actions)
        
        # Compute intrinsic rewards
        int_rewards = maddpg.get_intrinsic_rewards(next_obs)
        rewards = np.array([ext_rewards]) + \
                  cfg.int_reward_coeff * np.array([int_rewards])
                #   explo_pct_remaining * np.array([int_rewards])
        
        # Store experience in replay buffer
        replay_buffer.push(
            np.expand_dims(obs, axis=0), 
            np.array([np.expand_dims(a, axis=0) for a in actions]), 
            rewards, 
            np.array([next_obs]), 
            np.array([dones]))
        
        # Store step data
        ep_returns += rewards[0]
        ep_ext_returns += ext_rewards
        ep_int_returns += int_rewards
        ep_length += 1
        if any(dones):
            ep_success = True

        # If episode is finished
        if any(dones) or ep_length == cfg.episode_length:
            # Log episode data
            train_data_dict["Step"].append(step_i)
            train_data_dict["Episode return"].append(np.mean(ep_returns))
            train_data_dict["Episode extrinsic return"].append(
                np.mean(ep_ext_returns))
            train_data_dict["Episode intrinsic return"].append(
                np.mean(ep_int_returns))
            train_data_dict["Success"].append(int(ep_success))
            train_data_dict["Episode length"].append(ep_length)
            # Tensorboard
            logger.add_scalar(
                'agent0/episode_return', 
                train_data_dict["Episode return"][-1], 
                train_data_dict["Step"][-1])
            logger.add_scalar(
                'agent0/episode_ext_return', 
                train_data_dict["Episode extrinsic return"][-1], 
                train_data_dict["Step"][-1])
            logger.add_scalar(
                'agent0/episode_int_return', 
                train_data_dict["Episode intrinsic return"][-1], 
                train_data_dict["Step"][-1])
            # Reset the environment
            ep_returns = np.zeros(nb_agents)
            ep_ext_returns = np.zeros(nb_agents)
            ep_int_returns = np.zeros(nb_agents)
            ep_length = 0
            ep_success = False
            obs = env.reset()
            maddpg.reset_noise()
            maddpg.reset_noveld()
        else:
            obs = next_obs

        # Training
        if ((step_i + 1) % cfg.frames_per_update == 0 and
                len(replay_buffer) >= cfg.batch_size):
            maddpg.prep_training(device=device)
            samples = [replay_buffer.sample(
                            config.batch_size, cuda_device=device)
                       for _ in range(nb_agents)]
            vf_loss, pol_loss, nd_loss = maddpg.update(samples)
            # Log
            for a_i in range(nb_agents):
                logger.add_scalars(
                    'agent%i/losses' % a_i, 
                    {'vf_loss': vf_loss[a_i],
                     'pol_loss': pol_loss[a_i]},
                    #  'nd_loss': nd_loss[a_i]},
                    step_i)
            maddpg.update_all_targets()
            maddpg.prep_rollouts(device='cpu')

        # Evalutation
        if cfg.eval_every is not None and (step_i + 1) % cfg.eval_every == 0:
            eval_return, eval_success_rate, eval_ep_len = perform_eval_scenar(
                env, maddpg, eval_scenar, cfg.episode_length)
            eval_data_dict["Step"].append(step_i + 1)
            eval_data_dict["Mean return"].append(eval_return)
            eval_data_dict["Success rate"].append(eval_success_rate)
            eval_data_dict["Mean episode length"].append(eval_ep_len)

        # Save model
        if (step_i + 1) % cfg.save_interval == 0:
            os.makedirs(run_dir / 'incremental', exist_ok=True)
            maddpg.save(run_dir / 'incremental' / ('model_ep%i.pt' % (step_i)))
            maddpg.save(model_cp_path)
            maddpg.prep_rollouts(device='cpu')
    
    env.close()
    # Save model
    maddpg.save(model_cp_path)
    # Log Tensorboard
    logger.export_scalars_to_json(str(log_dir / 'summary.json'))
    logger.close()
    # Save training and eval data
    train_df = pd.DataFrame(train_data_dict)
    train_df.to_csv(str(run_dir / 'training_data.csv'))
    if cfg.eval_every is not None:
        eval_df = pd.DataFrame(eval_data_dict)
        eval_df.to_csv(str(run_dir / 'evaluation_data.csv'))
    print("Model saved in dir", run_dir)




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
    # Model hyperparameters
    parser.add_argument("--hidden_dim", default=64, type=int)
    parser.add_argument("--lr", default=0.0007, type=float)
    parser.add_argument("--tau", default=0.01, type=float)
    parser.add_argument("--gamma", default=0.99, type=float)
    parser.add_argument("--shared_params", action='store_true')
    # NovelD
    parser.add_argument("--noveld_type", default="multi_agent", type=str, 
                        choices=["multi_agent", "per_agent", "multi+per_agent"])
    parser.add_argument("--embed_dim", default=16, type=int)
    parser.add_argument("--int_reward_coeff", default=0.1, type=float)
    # Cuda
    parser.add_argument("--cuda_device", default=None, type=str)

    config = parser.parse_args()

    run(config)