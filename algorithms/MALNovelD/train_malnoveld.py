import argparse
import os
import json
import torch
import numpy as np
import pandas as pd

from tensorboardX import SummaryWriter
from tqdm import tqdm, trange

from utils.buffer import ReplayBuffer, LanguageBuffer
from utils.make_env import get_paths, load_scenario_config
from utils.make_env_parser import make_env
from utils.eval import perform_eval_scenar
from utils.decay import EpsilonDecay
from model.malnoveld import MALNovelD


def run(cfg):
    torch.autograd.set_detect_anomaly(True)
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
    env, parser = make_env(cfg, sce_conf, cfg.discrete_action)

    # Create model
    n_agents = sce_conf["nb_agents"]
    input_dim = env.observation_space[0].shape[0]
    if cfg.discrete_action:
        act_dim = env.action_space[0].n
    else:
        act_dim = env.action_space[0].shape[0]
    model = MALNovelD(
        input_dim, act_dim, cfg.embed_dim, n_agents, parser.vocab, cfg.lr,
        cfg.gamma, cfg.tau, cfg.temp, cfg.hidden_dim, cfg.context_dim, 
        cfg.init_explo_rate, cfg.noveld_lr, cfg.noveld_scale, 
        cfg.noveld_trade_off, cfg.discrete_action, cfg.shared_params,
        cfg.policy_algo
    )
    model.prep_rollouts(device=device)

    # Create data buffers
    replay_buffer = ReplayBuffer(
        cfg.buffer_length, 
        n_agents,
        [obsp.shape[0] for obsp in env.observation_space],
        [acsp.shape[0] if not cfg.discrete_action else acsp.n
            for acsp in env.action_space]
    )
    language_buffer = LanguageBuffer(cfg.buffer_length)

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
    print(f"    policy updates every {cfg.frames_per_policy_update} frames")
    print(f"    lnoveld updates every {cfg.frames_per_lnoveld_update} frames")
    print(f"    language updates every {cfg.frames_per_language_update} frames")
    print(f"    with seed {cfg.seed}")
    train_data_dict = {
        "Step": [],
        "Episode return": [],
        "Episode extrinsic return": [],
        "Episode intrinsic return": [],
        "Success": [],
        "Episode length": []
    }
    # Reset episode data and environment
    ep_returns = np.zeros(n_agents)
    ep_ext_returns = np.zeros(n_agents)
    ep_int_returns = np.zeros(n_agents)
    ep_length = 0
    ep_success = False
    # Reset environment and get first observations
    obs = env.reset()
    # Get first descriptions
    descr = parser.get_descriptions(obs, sce_conf)
    for step_i in trange(cfg.n_frames):
        # Compute and set exploration rate
        model.update_exploration_rate(eps_decay.get_explo_rate(step_i))

        # PERFORM STEP
        # Get actions
        actions = model.step(obs, descr, explore=True)
        actions = [a.squeeze().cpu().data.numpy() for a in actions]
        next_obs, ext_rewards, dones, _ = env.step(actions)

        # Compute intrinsic rewards
        next_descr = parser.get_descriptions(next_obs, sce_conf)
        int_rewards = model.get_intrinsic_rewards(next_obs, next_descr)

        # Compute final reward
        rewards = np.array([ext_rewards]) + \
                  cfg.int_reward_coeff * np.array([int_rewards])

        # Store experience in buffers
        replay_buffer.push(
            np.expand_dims(obs, axis=0), 
            np.array([np.expand_dims(a, axis=0) for a in actions]), 
            rewards, 
            np.array([next_obs]), 
            np.array([dones]))
        language_buffer.store(obs, descr)

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
            ep_returns = np.zeros(n_agents)
            ep_ext_returns = np.zeros(n_agents)
            ep_int_returns = np.zeros(n_agents)
            ep_length = 0
            ep_success = False
            obs = env.reset()
            model.reset()
        else:
            obs = next_obs
            descr = next_descr

        # Training
        if ((step_i + 1) % cfg.frames_per_policy_update == 0 and
                len(replay_buffer) >= cfg.batch_size):
            model.prep_training(device=device)
            # Policy training
            exp_samples = [replay_buffer.sample(
                                cfg.batch_size, cuda_device=device)
                           for _ in range(n_agents)]
            vf_losses, pol_losses = model.update_policy(exp_samples)
            log_dict = {
                "vf_loss": vf_losses[0],
                "pol_loss": pol_losses[0]
            }
            # L-NovelD training
            if (step_i + 1) % cfg.frames_per_lnoveld_update == 0:
                lnd_obs_loss, lnd_lang_loss = model.update_lnoveld()
                log_dict["lnd_obs_loss"] = lnd_obs_loss
                log_dict["lnd_lang_loss"] = lnd_lang_loss
            # Language learning
            if (step_i + 1) % cfg.frames_per_language_update == 0:
                lang_sample = language_buffer.sample(cfg.batch_size)
                clip_loss = model.update_language_modules(lang_sample)
                log_dict["clip_loss"] = clip_loss
            # Log
            logger.add_scalars(
                'agent0/losses', 
                log_dict,
                step_i)
            # lang_sample = language_buffer.sample(cfg.batch_size)
            # vf_loss, pol_loss, clip_loss, lnd_obs_loss, lnd_lang_loss = \
            #     model.update(exp_samples, lang_sample)
            # # Log
            # for a_i in range(n_agents):
            #     logger.add_scalars(
            #         'agent%i/losses' % a_i, 
            #         {'vf_loss': vf_loss[a_i],
            #          'pol_loss': pol_loss[a_i],
            #          'nd_loss': clip_loss,
            #          'nd_loss': lnd_obs_loss,
            #          'nd_loss': lnd_lang_loss},
            #         step_i)
            model.update_all_targets()
            model.prep_rollouts(device=device)
        
        # Evalutation
        if cfg.eval_every is not None and (step_i + 1) % cfg.eval_every == 0:
            eval_return, eval_success_rate, eval_ep_len = perform_eval_scenar(
                env, model, eval_scenar, cfg.episode_length)
            eval_data_dict["Step"].append(step_i + 1)
            eval_data_dict["Mean return"].append(eval_return)
            eval_data_dict["Success rate"].append(eval_success_rate)
            eval_data_dict["Mean episode length"].append(eval_ep_len)
        
        # Save model
        if (step_i + 1) % cfg.save_interval == 0:
            os.makedirs(run_dir / 'incremental', exist_ok=True)
            model.save(run_dir / 'incremental' / ('model_ep%i.pt' % (step_i)))
            model.save(model_cp_path)
            model.prep_rollouts(device=device)

    env.close()
    # Save model
    model.save(model_cp_path)
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
        default="algorithms/MALNovelD/scenarios/coop_push_scenario_parse.py")
    parser.add_argument("--model_name", type=str, default="TEST_malnoveld",
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
    parser.add_argument("--frames_per_policy_update", 
                        default=100, type=int)
    parser.add_argument("--frames_per_lnoveld_update", 
                        default=1000, type=int)
    parser.add_argument("--frames_per_language_update", 
                        default=1000, type=int)
    parser.add_argument("--batch_size", default=512, type=int,
                        help="Batch size for model training")
    parser.add_argument("--n_explo_frames", default=None, type=int,
        help="Number of frames where agents explore, if None then equal to n_frames")
    parser.add_argument("--explo_strat", default="sample", type=str)
    parser.add_argument("--init_explo_rate", default=1.0, type=float)
    parser.add_argument("--final_explo_rate", default=0.0, type=float)
    parser.add_argument("--epsilon_decay_fn", default="linear", type=str)
    parser.add_argument("--save_interval", default=100000, type=int)
    parser.add_argument("--int_reward_coeff", default=0.1, type=float)
    # Evalutation
    parser.add_argument("--eval_every", type=int, default=None)
    parser.add_argument("--eval_scenar_file", type=str, default=None)
    # Model hyperparameters
    parser.add_argument("--context_dim", default=16, type=int)
    parser.add_argument("--lr", default=0.0007, type=float)
    parser.add_argument("--temp", default=1.0, type=float)
    # Policy hyperparameters
    parser.add_argument("--policy_algo", default="maddpg", type=str)
    parser.add_argument("--hidden_dim", default=64, type=int)
    parser.add_argument("--tau", default=0.01, type=float)
    parser.add_argument("--gamma", default=0.99, type=float)
    parser.add_argument("--shared_params", action='store_true')
    # LNovelD
    parser.add_argument("--embed_dim", default=16, type=int)
    parser.add_argument("--noveld_lr", default=1e-4, type=float)
    parser.add_argument("--noveld_scale", default=0.5, type=float)
    parser.add_argument("--noveld_trade_off", default=1.0, type=float)
    # Observation Parser
    parser.add_argument("--chance_not_sent", default=0.1, type=float)
    parser.add_argument("--parser", default="basic", type=str, 
                        help="Available parsers are 'basic' and 'strat'")
    # Cuda
    parser.add_argument("--cuda_device", default=None, type=str)

    config = parser.parse_args()

    run(config)