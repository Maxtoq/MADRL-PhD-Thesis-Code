import os
import json
import numpy as np
import pandas as pd

from tqdm import trange

from src.utils.config import get_config
from src.utils.utils import set_seeds, set_cuda_device, load_args, render, save_frames, log_comm
from src.envs.make_env import make_env
from src.algo.lgmarl_diff import LanguageGroundedMARL



def run_eval(cfg):
    # Get pretrained stuff
    assert cfg.model_dir is not None, "Must provide model_dir"
    load_args(cfg, eval=True)
    if ',' in cfg.model_dir:
        paths = cfg.model_dir.split(',')
        pretrained_model_path = []
        for p in paths:
            pretrained_model_path.append(os.path.join(p, "model_ep.pt"))
            assert os.path.isfile(pretrained_model_path[-1]), "No model checkpoint found at" + str(pretrained_model_path[-1])
        print("Loading checkpoints from runs", paths)
    else:
        pretrained_model_path = os.path.join(cfg.model_dir, "model_ep.pt")
        assert os.path.isfile(pretrained_model_path), "No model checkpoint found at" + str(pretrained_model_path)
    print("Starting eval with config:")
    print(cfg)
    # cfg.comm_type = "language"
    # if cfg.comm_type == "perfect":
    #     cfg.comm_type = "language_sup"

    set_seeds(cfg.seed)

    # Set training device
    device = set_cuda_device(cfg)

    # Get eval scenarios
    init_positions = None
    if cfg.eval_scenario is not None:
        with open(cfg.eval_scenario, 'r') as f:
            init_positions = json.load(f)
    
    # Create train environment
    envs, parser = make_env(
        cfg, cfg.n_parallel_envs, init_positions=init_positions)

    log_dir = None
    if cfg.log_comm:
        log_dir = "results/comm_logs/" + cfg.model_dir.split("/")[-2] + ".csv"
    
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
        None,
        comm_eps_start=0.0)

    # Load params
    model.load(pretrained_model_path)

    print(f"Starting eval for {cfg.n_steps} frames")
    print(f"                  with {cfg.n_parallel_envs} parallel rollouts")
    print(f"                  with seed {cfg.seed}")

    obs = envs.reset()
    parsed_obs = parser.get_perfect_messages(obs)

    # img = render(cfg, envs)
    # if cfg.save_render:
    #     frames = [img]

    model.init_episode(obs, parsed_obs)
    model.prep_rollout(device)

    logs = {
        "obs": [],
        "messages": [],
        "actions": [],
        "perfect_messages": []
    }
    n_steps_per_update = cfg.n_parallel_envs * cfg.rollout_length
    for s_i in trange(0, cfg.n_steps, n_steps_per_update, ncols=0):
        for ep_s_i in range(cfg.rollout_length):
            # add_mess = interact(cfg)
            
            # Get action
            actions, agent_messages, _, comm_rewards \
                = model.act(deterministic=True)

            # Perform action and get reward and next obs
            next_obs, rewards, dones, infos = envs.step(actions)

            ll = model.model.lang_learner if type(model.model.lang_learner) != list \
                    else model.model.lang_learner[0]
            enc_perf_mess, _ = model.model.encode_perf_messages(parsed_obs, False)

            logs["obs"].append(obs)
            logs["messages"].append(agent_messages)
            logs["actions"].append(actions)
            logs["perfect_messages"].append(enc_perf_mess)

            obs = next_obs
            parsed_obs = parser.get_perfect_messages(obs)
            model.store_exp(obs, parsed_obs, rewards, dones)

            # count_returns += rewards.mean(-1)
            # img = render(cfg, envs)
            # if cfg.save_render:
            #     frames.append(img)

            # env_dones = dones.all(axis=1)
            # if True in env_dones:
            #     returns += list(count_returns[env_dones == True])
            #     count_returns *= (1 - env_dones)
        model.init_episode()

        print(np.concatenate(logs["obs"]).shape)
        exit()

        # if cfg.save_render:
        #     save_frames(frames, cfg.model_dir)
            
    envs.close()

    if cfg.log_comm:
        print("Saving comm logs to", log_dir)
        comm_logs = pd.DataFrame(comm)
        comm_logs.to_csv(log_dir)

    print("Done", len(returns), "complete episodes.")
    print("Returns", returns)
    print("Mean return:", sum(returns) / len(returns))

    return sum(returns) / len(returns)

if __name__ == '__main__':
    # Load config
    parser = get_config()
    cfg = parser.parse_args()

    run_eval(cfg)
    