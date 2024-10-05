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
    interact_logs = {
        "Message":[],
        "A0a0": [],
        "A0a1": [],
        "A0a2": [],
        "A0a3": [],
        "A0a4": [],
        "A1a0": [],
        "A1a1": [],
        "A1a2": [],
        "A1a3": [],
        "A1a4": [],
        "A2a0": [],
        "A2a1": [],
        "A2a2": [],
        "A2a3": [],
        "A2a4": [],
        "A3a0": [],
        "A3a1": [],
        "A3a2": [],
        "A3a3": [],
        "A3a4": []
    }
    # for sp in sent_pos_test_pairs.copy():
    #     sent_pos_test_pairs.append(([], sp[1]))
    # interact_log = pd.DataFrame({
    #     "Message": [sp[0] for sp in sent_pos_test_pairs],
    #     "Init_pos": [sp[1] for sp in sent_pos_test_pairs]})
    
    # Create train environment
    cfg.env_name = "magym_Empty"
    cfg.n_parallel_envs = 100
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
        for e_i in trange(1000):
            obs = envs.reset()
            parsed_obs = parser.get_perfect_messages(obs)

            model.init_episode(obs, parsed_obs)
            for ep_s_i in range(cfg.rollout_length):
                # Get action
                actions, agent_messages, _, comm_rewards \
                    = model.act(
                        deterministic=True, 
                        lang_input=input_message)# if ep_s_i == 0 else None)

                performed_actions.append(actions)

                # Perform action and get reward and next obs
                next_obs, rewards, dones, infos = envs.step(actions)

                obs = next_obs
                parsed_obs = parser.get_perfect_messages(obs)
                model.store_exp(obs, parsed_obs, rewards, dones)

                # render(cfg, envs)
            # Reset buffer for new episode
            # model.init_episode()
        
        
        agent_actions = np.concatenate(performed_actions).squeeze(-1).T
        interact_logs["Message"].append(tm)
        for ag_i in range(4):
            for a_i in range(5):
                action_count = (agent_actions[ag_i] == a_i).sum()
                interact_logs['A' + str(ag_i) + 'a' + str(a_i)].append(
                    action_count / agent_actions.shape[-1])
    envs.close()

    # print(interact_log)
    df = pd.DataFrame(interact_logs)
    df.to_csv("./results/data/lamarl_interact/" + cfg.model_dir.split("/")[-2] + cfg.model_dir[-1] + "_random.csv")

if __name__ == '__main__':
    # Load config
    parser = get_config()
    cfg = parser.parse_args()

    run_eval(cfg)
    