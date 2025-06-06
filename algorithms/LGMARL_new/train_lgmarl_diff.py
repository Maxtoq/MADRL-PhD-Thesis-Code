import os

from tqdm import trange

from src.utils.config import get_config
from src.utils.utils import set_seeds, set_cuda_device, load_args
from src.log.log_wandb import Logger
from src.log.util import get_paths, write_params
from src.envs.make_env import make_env
from src.algo.lgmarl_diff import LanguageGroundedMARL


def run():
     # Load config
    argparse = get_config()
    cfg = argparse.parse_args()

    # Get paths for saving logs and model
    run_dir, log_dir = get_paths(cfg)
    print("Saving model in dir", run_dir)

    # Load pretrained checkpoint if needed
    if cfg.model_dir is not None:
        # In case of adapt, go take the model corresponding to the run number
        if os.path.basename(cfg.model_dir)[:3] != "run":
            cfg.model_dir = os.path.join(cfg.model_dir, os.path.basename(run_dir))
        print(cfg.model_dir)

        # Get pretrained stuff
        steps_done = load_args(cfg)

        pretrained_model_path = os.path.join(cfg.model_dir, "model_ep.pt")
        assert os.path.isfile(pretrained_model_path), "No model checkpoint provided."

        comm_eps_start = cfg.FT_comm_eps_start
        cfg.comm_eps_nsteps = cfg.FT_freeze_lang_after_n

        print("Starting from pretrained model with config:")
        print(cfg)
    else:
        comm_eps_start = 1.0
        comm_eps_nsteps = cfg.comm_eps_nsteps

    # Set training step (if continue previous run)
    if cfg.continue_run:
        start_step = steps_done
        if cfg.n_steps <= start_step:
            print("No need to train more.")
            exit()
    else:
        start_step = 0

    write_params(run_dir, cfg)

    # Init logger
    logger = Logger(cfg, log_dir, start_step)

    set_seeds(cfg.seed)

    # Set training device
    device = set_cuda_device(cfg)
    
    # Create train environment
    envs, parser = make_env(cfg, cfg.n_parallel_envs)

    # Create model
    n_agents = envs.n_agents
    obs_space = envs.observation_space
    shared_obs_space = envs.shared_observation_space
    act_space = envs.action_space
    model = LanguageGroundedMARL(
        cfg, 
        n_agents, 
        obs_space, 
        shared_obs_space, 
        act_space,
        parser, 
        device,
        log_dir,
        comm_eps_start,
        cfg.comm_eps_nsteps)

    # Load params
    if cfg.model_dir is not None:
        model.load(pretrained_model_path)

    # Start training
    n_steps_per_update = cfg.n_parallel_envs * cfg.rollout_length
    print(f"Starting training for {cfg.n_steps - start_step} frames")
    print(f"                  with {cfg.n_parallel_envs} parallel rollouts")
    print(f"                  updates every {n_steps_per_update} frames")
    print(f"                  with seed {cfg.seed}")
    # Reset env
    last_save_step = 0
    last_eval_step = 0
    last_scale_step = 0
    obs = envs.reset()
    parsed_obs = parser.get_perfect_messages(obs)
    model.init_episode(obs, parsed_obs)
    for s_i in trange(start_step, cfg.n_steps, n_steps_per_update, ncols=0):
        model.prep_rollout(device)

        for ep_s_i in range(cfg.rollout_length):
            # Perform step
            # Get action
            actions, broadcasts, agent_messages, comm_rewards \
                = model.act()

            # Perform action and get reward and next obs
            obs, rewards, dones, infos = envs.step(actions)

            # env_dones = dones.all(axis=1)
            # if True in env_dones:
            #     model.reset_context(env_dones)

            # Log rewards
            logger.count_returns(s_i, rewards, dones)

            # Insert data into policy buffer
            parsed_obs = parser.get_perfect_messages(obs)
            model.store_exp(obs, parsed_obs, rewards, dones)

        # Training
        train_lang = True
        if cfg.FT_freeze_lang_after_n is not None and s_i >= cfg.FT_freeze_lang_after_n:
            train_lang = False
        train_losses = model.train(
            s_i + n_steps_per_update,
            train_lang=train_lang)

        # Log train data
        logger.log_losses(train_losses, s_i + n_steps_per_update)

        # Reset buffer for new episode
        model.init_episode()

        # Save
        if s_i + n_steps_per_update - last_save_step > cfg.save_interval:
            last_save_step = s_i + n_steps_per_update
            if cfg.save_increments:
                model.save(run_dir / "incremental" / f"model_ep{last_save_step}.pt")

        if cfg.magym_scaleenv_after_n is not None and \
                s_i + n_steps_per_update - last_scale_step >= cfg.magym_scaleenv_after_n:
            last_scale_step = s_i + n_steps_per_update

            envs.close()

            cfg.magym_env_size += 3
            print("Scaling envs...", end=' ')
            while True:
                try:
                    envs, parser = make_env(cfg, cfg.n_parallel_envs)
                    break
                except OSError:
                    pass
            print("done.")
            write_params(run_dir, cfg)

            obs = envs.reset()
            parsed_obs = parser.get_perfect_messages(obs)
            model.init_episode(obs, parsed_obs)
            
    envs.close()
    # Save model and training data
    logger.save_n_close()
    model.save(run_dir / "model_ep.pt")

if __name__ == '__main__':
    run()
