import argparse
import os
import git
import json
import time
import torch
import random
import numpy as np
import pandas as pd

from tensorboardX import SummaryWriter
from tqdm import tqdm

from model.modules.qmix import QMIX
from model.modules.qmix_noveld import QMIX_MANovelD, QMIX_PANovelD, QMIX_MALNovelD
from utils.buffer import RecReplayBuffer
from utils.make_env import get_paths, load_scenario_config, make_env, make_env_parser
from utils.eval import perform_eval_scenar
from utils.decay import EpsilonDecay

def run(cfg):
    # Get paths for saving logs and model
    run_dir, model_cp_path, log_dir = get_paths(cfg)
    print("Saving model in dir", run_dir)

    # Save args in txt file
    with open(os.path.join(run_dir, 'args.txt'), 'w') as f:
        f.write(str(time.time()) + '\n')
        commit_hash = git.Repo(
            search_parent_directories=True).head.object.hexsha
        f.write(
            "Running train_qmix.py at git commit " + str(commit_hash) + '\n')
        f.write("Parameters:\n")
        f.write(json.dumps(vars(cfg), indent=4))

    # Init summary writer
    logger = SummaryWriter(str(log_dir))

    # Load scenario config
    sce_conf = load_scenario_config(cfg, run_dir)

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    # Set training device
    if torch.cuda.is_available():
        if cfg.cuda_device is None:
            device = 'cuda'
        else:
            device = torch.device(cfg.cuda_device)
    else:
        device = 'cpu'

    # Create environment
    if "lnoveld" in cfg.model_type:
        env, parser = make_env_parser(
            cfg.env_path, sce_conf, discrete_action=True)
        lnoveld = True
    else:
        env = make_env(cfg.env_path, sce_conf, discrete_action=True)
        lnoveld = False

    # Create model
    nb_agents = sce_conf["nb_agents"]
    obs_dim = env.observation_space[0].shape[0]
    act_dim = env.action_space[0].n
    if cfg.model_type == "qmix":
        qmix = QMIX(nb_agents, obs_dim, act_dim, cfg.lr, cfg.gamma, cfg.tau, 
            cfg.hidden_dim, cfg.shared_params, cfg.init_explo_rate,
            cfg.max_grad_norm, device)
    elif cfg.model_type == "qmix_manoveld":
        qmix = QMIX_MANovelD(nb_agents, obs_dim, act_dim, cfg.lr, 
            cfg.gamma, cfg.tau, cfg.hidden_dim, cfg.shared_params, 
            cfg.init_explo_rate, cfg.max_grad_norm, device,
            cfg.embed_dim, cfg.nd_lr, cfg.nd_scale_fac, cfg.nd_hidden_dim)
    elif cfg.model_type == "qmix_panoveld":
        qmix = QMIX_PANovelD(nb_agents, obs_dim, act_dim, cfg.lr, 
            cfg.gamma, cfg.tau, cfg.hidden_dim, cfg.shared_params, 
            cfg.init_explo_rate, cfg.max_grad_norm, device,
            cfg.embed_dim, cfg.nd_lr, cfg.nd_scale_fac, cfg.nd_hidden_dim)
    elif cfg.model_type == "qmix_malnoveld":
        qmix = QMIX_MALNovelD(nb_agents, obs_dim, act_dim, cfg.lr, 
            parser.vocab, cfg.gamma, cfg.tau, cfg.hidden_dim, 
            cfg.shared_params, cfg.init_explo_rate, cfg.max_grad_norm, device,
            cfg.embed_dim, cfg.nd_lr, cfg.nd_scale_fac, cfg.nd_hidden_dim)
    else:
        print("ERROR: bad model type.")
    qmix.prep_rollouts(device=device)

    # Create replay buffer
    buffer = RecReplayBuffer(
        cfg.buffer_length, cfg.episode_length, nb_agents, obs_dim, act_dim)

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
        "Episode return": [],
        "Episode extrinsic return": [],
        "Episode intrinsic return": [],
        "Episode intrinsic return 2": [],
        "Success": [],
        "Episode length": []
    }
    if cfg.save_descriptions:
        saved_descrs = []
    # Reset episode data and environment
    ep_step_i = 0
    ep_ext_returns = np.zeros(nb_agents)
    ep_int_returns = np.zeros(nb_agents)
    ep_int_returns2 = np.zeros(nb_agents)
    ep_success = False
    obs = env.reset()
    if lnoveld:
        # Get first descriptions
        descr = parser.get_descriptions(obs)
    # Init episode data for saving in replay buffer
    ep_obs, ep_shared_obs, ep_acts, ep_rews, ep_dones = \
        buffer.init_episode_arrays()
    # Get initial last actions and hidden states
    last_actions, qnets_hidden_states = qmix.get_init_model_inputs()
    for step_i in tqdm(range(cfg.n_frames)):
        qmix.set_explo_rate(eps_decay.get_explo_rate(step_i))

        # Get actions
        if lnoveld:
            actions, qnets_hidden_states = qmix.get_actions(
                obs, last_actions, qnets_hidden_states, descr, explore=True)
        else:
            actions, qnets_hidden_states = qmix.get_actions(
                obs, last_actions, qnets_hidden_states, explore=True)
        last_actions = actions
        actions = [a.cpu().squeeze().data.numpy() for a in actions]
        next_obs, ext_rewards, dones, _ = env.step(actions)

        # Compute intrinsic rewards
        if "noveld" in cfg.model_type:
            if lnoveld:
                next_descr = parser.get_descriptions(next_obs)
                int_rewards, lnd_obs_rew, lnd_lang_rew = \
                    qmix.get_intrinsic_rewards(next_obs, next_descr)
            else:
                int_rewards = qmix.get_intrinsic_rewards(next_obs)
            rewards = np.array([ext_rewards]) + \
                      cfg.int_reward_coeff * np.array([int_rewards])
            rewards = rewards.T
        else:
            int_rewards = [0.0, 0.0]
            rewards = np.vstack(ext_rewards)

        # Save experience for replay buffer
        ep_obs[ep_step_i, 0, :] = np.stack(obs)
        ep_shared_obs[ep_step_i, 0, :] = np.tile(np.concatenate(obs), (2, 1))
        ep_acts[ep_step_i, 0, :] = np.stack(actions)
        ep_rews[ep_step_i, 0, :] = rewards
        ep_dones[ep_step_i, 0, :] = np.vstack(dones)

        ep_ext_returns += ext_rewards
        if lnoveld:
            ep_int_returns += lnd_obs_rew
            ep_int_returns2 += lnd_lang_rew
        else:
            ep_int_returns += int_rewards
        if any(dones):
            ep_success = True

        if cfg.save_descriptions:
            saved_descrs.append(
                {"step": step_i, "obs": list(obs), "descr": descr})
        
        # Check for end of episode
        if ep_success or ep_step_i + 1 == cfg.episode_length:
            # Store next state observations for last step
            ep_obs[ep_step_i + 1, 0, :] = np.stack(next_obs)
            ep_shared_obs[ep_step_i + 1, 0, :] = np.tile( 
                np.concatenate(next_obs), (2, 1))
            # Store episode in replay buffer
            buffer.store(ep_obs, ep_shared_obs, ep_acts, ep_rews, ep_dones)
            # Log training data
            train_data_dict["Step"].append(step_i)
            train_data_dict["Episode return"].append(
                np.sum(ep_rews) / nb_agents)
            train_data_dict["Episode extrinsic return"].append(
                np.mean(ep_ext_returns))
            train_data_dict["Episode intrinsic return"].append(
                np.mean(ep_int_returns))
            train_data_dict["Episode intrinsic return 2"].append(
                np.mean(ep_int_returns2))
            train_data_dict["Success"].append(int(ep_success))
            train_data_dict["Episode length"].append(ep_step_i + 1)
            # Log Tensorboard
            logger.add_scalar(
                'agent0/episode_return', 
                train_data_dict["Episode return"][-1], 
                train_data_dict["Step"][-1])
            logger.add_scalar(
                'agent0/episode_ext_return', 
                train_data_dict["Episode extrinsic return"][-1], 
                train_data_dict["Step"][-1])
            if lnoveld:
                int_returns_dict = {
                    "lnd_obs_return": np.mean(ep_int_returns),
                    "lnd_lang_return": np.mean(ep_int_returns2)}
                logger.add_scalars(
                    'agent0/episode_int_return', 
                    int_returns_dict,
                    train_data_dict["Step"][-1])
            else:
                logger.add_scalar(
                    'agent0/episode_int_return', 
                    train_data_dict["Episode intrinsic return"][-1], 
                    train_data_dict["Step"][-1])
            # Reset episode data
            ep_ext_returns = np.zeros(nb_agents)
            ep_int_returns = np.zeros(nb_agents)
            ep_int_returns2 = np.zeros(nb_agents)
            ep_step_i = 0
            ep_success = False
            # Init episode data for saving in replay buffer
            ep_obs, ep_shared_obs, ep_acts, ep_rews, ep_dones = \
                buffer.init_episode_arrays()
            # Get initial last actions and hidden states
            last_actions, qnets_hidden_states = qmix.get_init_model_inputs()
            # Reset environment
            obs = env.reset()
            if "noveld" in cfg.model_type:
                qmix.reset_noveld()
        else:
            ep_step_i += 1
            obs = next_obs
            if lnoveld:
                descr = next_descr

        # Training
        if ((step_i + 1) % cfg.frames_per_update == 0 and
                len(buffer) >= cfg.batch_size):
            qmix.prep_training(device=device)
            # Get samples
            sample_batch = buffer.sample(cfg.batch_size, device)
            # Train
            losses = qmix.train_on_batch(sample_batch)
            if cfg.model_type == "qmix":
                loss_dict = {"qtot_loss": losses}
            elif cfg.model_type in ["qmix_manoveld", "qmix_panoveld"]:
                loss_dict = {
                    "qtot_loss": losses[0],
                    "nd_loss": losses[1]}
            elif cfg.model_type == "qmix_malnoveld":
                loss_dict = {
                    "qtot_loss": losses[0],
                    "lnd_obs_loss": losses[1],
                    "lnd_lang_loss": losses[2]}
            # Log
            logger.add_scalars('agent0/losses', loss_dict, step_i)
            qmix.update_all_targets()
            qmix.prep_rollouts(device=device)
            
        # Evaluation
        if cfg.eval_every is not None and (step_i + 1) % cfg.eval_every == 0:
            eval_return, eval_success_rate, eval_ep_len = perform_eval_scenar(
                env, qmix, eval_scenar, cfg.episode_length, recurrent=True)
            eval_data_dict["Step"].append(step_i + 1)
            eval_data_dict["Mean return"].append(eval_return)
            eval_data_dict["Success rate"].append(eval_success_rate)
            eval_data_dict["Mean episode length"].append(eval_ep_len)
            # Save eval data
            eval_df = pd.DataFrame(eval_data_dict)
            eval_df.to_csv(str(run_dir / 'evaluation_data.csv'))

        # Save model
        if (step_i + 1) % cfg.save_interval == 0:
            os.makedirs(run_dir / 'incremental', exist_ok=True)
            qmix.save(run_dir / 'incremental' / ('model_ep%i.pt' % (step_i)))
            qmix.save(model_cp_path)
            qmix.prep_rollouts(device=device)
            # Save training data
            train_df = pd.DataFrame(train_data_dict)
            train_df.to_csv(str(run_dir / 'training_data.csv'))

    env.close()
    # Save model
    qmix.save(model_cp_path)
    # Log Tensorboard
    logger.export_scalars_to_json(str(log_dir / 'summary.json'))
    logger.close()
    # Save training and eval data
    train_df = pd.DataFrame(train_data_dict)
    train_df.to_csv(str(run_dir / 'training_data.csv'))
    if cfg.eval_every is not None:
        eval_df = pd.DataFrame(eval_data_dict)
        eval_df.to_csv(str(run_dir / 'evaluation_data.csv'))
    if cfg.save_descriptions:
        with open(str(run_dir / "descriptions.json"), 'w') as f:
            json.dump(saved_descrs, f)
    print("Model saved in dir", run_dir)



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
    parser.add_argument("--n_frames", default=2500, type=int,
                        help="Number of training frames to perform")
    parser.add_argument("--buffer_length", default=5000, type=int,
                        help="Max number of episodes stored in replay buffer.")
    parser.add_argument("--frames_per_update", default=100, type=int)
    parser.add_argument("--batch_size", default=32, type=int,
                        help="Number of episodes to sample from replay buffer for training.")
    parser.add_argument("--save_interval", default=100000, type=int)
    # Exploration
    parser.add_argument("--n_explo_frames", default=None, type=int,
                        help="Number of frames where agents explore, if None then equal to n_frames")
    parser.add_argument("--init_explo_rate", default=1.0, type=float)
    parser.add_argument("--final_explo_rate", default=0.0, type=float)
    parser.add_argument("--epsilon_decay_fn", default="linear", type=str,
                        choices=["linear", "exp"])
    # Evalutation
    parser.add_argument("--eval_every", type=int, default=None)
    parser.add_argument("--eval_scenar_file", type=str, default=None)
    # Model hyperparameters
    parser.add_argument("--model_type", default="qmix", type=str, 
                        choices=["qmix", "qmix_manoveld", "qmix_panoveld", "qmix_malnoveld"])
    parser.add_argument("--hidden_dim", default=64, type=int)
    parser.add_argument("--lr", default=0.0007, type=float)
    parser.add_argument("--tau", default=0.005, type=float)
    parser.add_argument("--gamma", default=0.99, type=float)
    parser.add_argument("--shared_params", action='store_true')
    parser.add_argument("--max_grad_norm", type=float, default=0.5,
                        help='Max norm of gradients (default: 0.5)')
    # NovelD
    parser.add_argument("--embed_dim", default=16, type=int)
    parser.add_argument("--nd_lr", default=1e-4, type=float)
    parser.add_argument("--nd_scale_fac", default=0.5, type=float)
    parser.add_argument("--nd_hidden_dim", default=64, type=int)
    parser.add_argument("--int_reward_coeff", default=0.1, type=float)
    # Language
    parser.add_argument("--save_descriptions", action="store_true")
    # Cuda
    parser.add_argument("--cuda_device", default=None, type=str)

    config = parser.parse_args()

    run(config)