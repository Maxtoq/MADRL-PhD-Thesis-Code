import os
import json
import time
import random
import numpy as np
import pandas as pd

from tqdm import trange

from src.utils.config import get_config
from src.utils.utils import set_seeds, set_cuda_device, load_args
from src.envs.make_env import make_env
from src.algo.lgmarl_diff import LanguageGroundedMARL

def render(cfg, envs):
    if cfg.use_render and cfg.n_parallel_envs < 50:
        envs.render("human")
    if cfg.render_wait_input:
        input()
    else:
        time.sleep(0.1)

def interact(cfg):
    if cfg.interact:
        add_mess = input("Input message:")
        if len(add_mess):
            add_mess = add_mess.split(" ")
        else:
            add_mess = []
    else:
        add_mess = None
    return add_mess

def log_comm(comm_tab, lang_learner, gen_mess, perf_mess):
    dec_mess = lang_learner.word_encoder.decode_batch(
        gen_mess.reshape(gen_mess.shape[0] * gen_mess.shape[1], -1))
    for e_i in range(gen_mess.shape[0]):
        for a_i in range(gen_mess.shape[1]):
            # print(dec_mess[e_i * gen_mess.shape[1] + a_i], perf_mess[e_i][a_i])
            comm_tab.append({
                "Generated_Message": dec_mess[e_i * gen_mess.shape[1] + a_i], 
                "Perfect_Message": perf_mess[e_i][a_i]})



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

    init_pos = []
    min_center = cfg.magym_env_size // 3
    for i in range(min_center, 2 * min_center):
        for j in range(min_center, 2 * min_center):
            i_pos = (i, j)
            for x in range(min_center, 2 * min_center):
                for y in range(min_center, 2 * min_center):
                    if (x, y) != i_pos: 
                        init_pos.append((i_pos, (x, y)))
    
    # Create train environment
    cfg.env_name = "magym_Empty"
    if len(init_pos) > 250:
        cfg.n_parallel_envs = 200
    else:
        cfg.n_parallel_envs = len(init_pos)
    envs, parser = make_env(
        cfg, cfg.n_parallel_envs)
    
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

    obs = envs.reset()
    parsed_obs = parser.get_perfect_messages(obs)

    render(cfg, envs)

    model.init_episode(obs, parsed_obs)
    model.prep_rollout(device)
    # n_steps_per_update = cfg.n_parallel_envs * cfg.rollout_length
    # for s_i in range(cfg.n_eval_runs):
    for tm in test_messages:
        print(tm)
        input_message = [[tm] for _ in range(cfg.n_parallel_envs)]
        performed_actions = []

        for i in range(0, len(init_pos), cfg.n_parallel_envs):
            print(i, i + cfg.n_parallel_envs)
            obs = envs.reset(init_pos[i:i + cfg.n_parallel_envs])
            parsed_obs = parser.get_perfect_messages(obs)

            model.init_episode(obs, parsed_obs)
            for ep_s_i in range(cfg.rollout_length):
                # Get action
                actions, agent_messages, _, comm_rewards \
                    = model.act(
                        deterministic=True, 
                        lang_input=input_message) # if ep_s_i == 0 else None)

                performed_actions.append(actions)

                # Perform action and get reward and next obs
                next_obs, rewards, dones, infos = envs.step(actions)

                obs = next_obs
                parsed_obs = parser.get_perfect_messages(obs)
                model.store_exp(obs, parsed_obs, rewards, dones)
        
        
        agent_actions = np.concatenate(performed_actions).squeeze(-1).T
        interact_logs["Message"].append(tm)
        for ag_i in range(cfg.magym_n_agents):
            for a_i in range(5):
                action_count = (agent_actions[ag_i] == a_i).sum()
                interact_logs[f"A{ag_i}a{a_i}"].append(action_count)#  / agent_actions.shape[-1])
    envs.close()

    # print(interact_log)
    df = pd.DataFrame(interact_logs)
    df.to_csv("./results/data/lamarl_interact/" + cfg.model_dir.split("/")[-2] + cfg.model_dir[-1] + "_center.csv")

if __name__ == '__main__':
    # Load config
    parser = get_config()
    cfg = parser.parse_args()

    run_eval(cfg)
    