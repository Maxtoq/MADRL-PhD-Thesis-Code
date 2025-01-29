import os
import json
import time
import random
import numpy as np
import pandas as pd

from tqdm import trange

from src.utils.config import get_config
from src.utils.utils import set_seeds, set_cuda_device, load_args, render, save_frames
from src.envs.make_env import make_env
from src.algo.lgmarl_diff import LanguageGroundedMARL



def run_eval(cfg):
    # Get pretrained stuff
    assert cfg.model_dir is not None, "Must provide model_dir"
    load_args(cfg, eval=True)
    pretrained_model_path = os.path.join(cfg.model_dir, "model_ep.pt")
    assert os.path.isfile(pretrained_model_path), "No model checkpoint found at" + str(pretrained_model_path)
    print("Starting eval with config:")
    print(cfg)

    set_seeds(cfg.seed)

    # Set device
    device = set_cuda_device(cfg)

    # Get eval scenarios
    init_positions = None
    if cfg.eval_scenario is not None:
        with open(cfg.eval_scenario, 'r') as f:
            init_positions = json.load(f)

    test_messages = [
        ["Prey", "North"],
        ["Prey", "South"],
        ["Prey", "East"],
        ["Prey", "West"],
        ["Prey", "North", "West"],
        ["Prey", "North", "East"],
        ["Prey", "South", "West"],
        ["Prey", "South", "East"],
        []
    ]
    interact_logs = { "Message":[] }
    for a_i in range(cfg.magym_n_agents):
        interact_logs[f"A{a_i}a0"] = []
        interact_logs[f"A{a_i}a1"] = []
        interact_logs[f"A{a_i}a2"] = []
        interact_logs[f"A{a_i}a3"] = []
        interact_logs[f"A{a_i}a4"] = []

    # init_pos = []
    # min_center = cfg.magym_env_size // 3
    # for i in range(min_center, 2 * min_center):
    #     for j in range(min_center, 2 * min_center):
    #         i_pos = (i, j)
    #         for x in range(min_center, 2 * min_center):
    #             for y in range(min_center, 2 * min_center):
    #                 if (x, y) != i_pos: 
    #                     init_pos.append((i_pos, (x, y)))
    # # init_pos = [((4, 4), (3, 4)), ((4, 4), (3, 4))]
    
    # # Create train environment
    # assert "Empty" in cfg.env_name, "should use empty env"
    # if len(init_pos) > 250:
    #     cfg.n_parallel_envs = 200
    # else:
    #     cfg.n_parallel_envs = len(init_pos)
    envs, parser = make_env(cfg)
    
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
        comm_eps_start=0.0,
        block_comm=True)

    # Load params
    model.load(pretrained_model_path)  
    model.prep_rollout()  

    # model.init_episode(obs, parsed_obs)
    # model.prep_rollout(device)
    # n_steps_per_update = cfg.n_parallel_envs * cfg.rollout_length
    # for s_i in range(cfg.n_eval_runs):
    for tm in test_messages:
        print(tm)
        input_message = [[tm] for _ in range(cfg.n_parallel_envs)]
        performed_actions = []

        # for i in range(0, len(init_pos), cfg.n_parallel_envs):
            # print(i, i + cfg.n_parallel_envs)
        obs = envs.reset()#init_pos[i:i + cfg.n_parallel_envs])
        parsed_obs = parser.get_perfect_messages(obs)

        if cfg.save_render:
            frames = []
            frames.append(render(cfg, envs)[1])

        model.init_episode(obs, parsed_obs)
        for ep_s_i in range(cfg.rollout_length):
            # Get action
            actions, agent_messages, _, comm_rewards \
                = model.act(
                    deterministic=True, 
                    lang_input=input_message if ep_s_i == 0 else None)

            performed_actions.append(actions)

            # Perform action and get reward and next obs
            next_obs, rewards, dones, infos = envs.step(actions)

            obs = next_obs
            parsed_obs = parser.get_perfect_messages(obs)
            model.store_exp(obs, parsed_obs, rewards, dones)

            if cfg.save_render:
                frames.append(render(cfg, envs)[1])

        if cfg.save_render:
            save_frames(frames, cfg.model_dir)
        
        
        agent_actions = np.concatenate(performed_actions).squeeze(-1).T
        interact_logs["Message"].append(tm)
        for ag_i in range(cfg.magym_n_agents):
            for a_i in range(5):
                action_count = (agent_actions[ag_i] == a_i).sum()
                interact_logs[f"A{ag_i}a{a_i}"].append(action_count)#  / agent_actions.shape[-1])
    envs.close()

    # print(interact_log)
    df = pd.DataFrame(interact_logs)
    df.to_csv("./results/data/lamarl_interact/" + cfg.model_dir.split("/")[-2] + cfg.model_dir[-1] + "_center_onestep.csv")

if __name__ == '__main__':
    # Load config
    parser = get_config()
    cfg = parser.parse_args()

    run_eval(cfg)
    