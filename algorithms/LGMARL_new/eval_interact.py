import os
import json
import time
import torch
import random
import itertools
import numpy as np
import pandas as pd

from tqdm import trange

from src.utils.config import get_config
from src.utils.utils import set_seeds, set_cuda_device, load_args
from src.envs.make_env import make_env
from src.algo.lgmarl_diff import LanguageGroundedMARL

def render(cfg, envs):
    if cfg.use_render:
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

def run_eval(cfg, comm_prob, n_eval_runs):
    set_seeds(cfg.seed)

    # Get pretrained stuff
    assert cfg.model_dir is not None, "Must provide model_dir"
    load_args(cfg, eval=True)
    pretrained_model_path = os.path.join(cfg.model_dir, "model_ep.pt")
    assert os.path.isfile(pretrained_model_path), "No model checkpoint found at" + str(pretrained_model_path)
    # print("Starting eval with config:")
    # print(cfg)
    # cfg.comm_type = "language"
    # if cfg.comm_type == "perfect":
    #     cfg.comm_type = "language_sup"


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
        # block_comm=comm_prob != -1.0)

    # Load params
    model.load(pretrained_model_path)
    model.prep_rollout(device)

    all_returns = np.array([])
    for e_i in trange(n_eval_runs):
        obs, prey_pos = envs.reset()

        parsed_obs = parser.get_perfect_messages(obs)
        parsed_prey_pos = parser.get_prey_pos(prey_pos)

        model.init_episode(obs, parsed_obs)

        count_returns = np.zeros(cfg.n_parallel_envs)
        returns = np.zeros(cfg.n_parallel_envs)
        env_dones = np.zeros(cfg.n_parallel_envs)
        for ep_s_i in range(0, cfg.episode_length):
            # add_mess = interact(cfg)
            # Lang input
            send_comm = random.random()
            # print(parsed_obs)
            # print(parsed_prey_pos)
            # exit()
            if send_comm < comm_prob:
                lang_input = parsed_prey_pos
            else:
                lang_input = None
            
            # Get action
            actions, agent_messages, _, comm_rewards \
                = model.act(deterministic=True, lang_input=lang_input)

            # Perform action and get reward and next obs
            obs, prey_pos, rewards, dones, infos = envs.step(actions)


            if cfg.n_parallel_envs == 1:
                ll = model.model.lang_learner if type(model.model.lang_learner) != list \
                    else model.model.lang_learner[0]
                decoded_messages = ll.word_encoder.decode_batch(
                    agent_messages.squeeze(0))
            
                print(f"\nStep #{ep_s_i + 1}")
                print("Observations", obs)
                print("Perfect Messages", parsed_obs)
                print("Agent Messages", decoded_messages)
                print("Actions (t-1)", actions)
                print("Rewards", rewards)
                print("Communication Rewards", comm_rewards)

            parsed_obs = parser.get_perfect_messages(obs)
            parsed_prey_pos = parser.get_prey_pos(prey_pos)

            model.store_exp(obs, parsed_obs, rewards, dones)

            count_returns += rewards.mean(-1)
            render(cfg, envs)

            dones = dones.all(axis=1)
            new_dones = dones * (1 - env_dones) == True
            if True in new_dones:
                env_dones += new_dones

                returns[new_dones] = count_returns[new_dones]
                # count_returns *= (1 - env_dones)
                if ep_s_i == cfg.rollout_length - 1:
                    break
                # print(f"ENV DONE: {int(sum(env_dones))} / {cfg.n_parallel_envs}")
                if env_dones.all():
                    break
        all_returns = np.concatenate((all_returns, returns))
            
    envs.close()

    return all_returns

if __name__ == '__main__':
    # Load config
    parser = get_config()
    cfg = parser.parse_args()

    # random.seed(cfg.seed)
    cfg.seed = 0

    comm_probs = [
        0.05, 0.1, 0.2, 0.5, 0.8, 1.0]
    results = {
        "Comm prob": [],
        "Mean return": [],
        "Std": [],
        "Median return": [],
        "Success rate": [],
        "Success std": []
    }
    log_file_path = '/'.join(cfg.model_dir.split(',')[0].split('/')[:-1]) + "/interact2.csv"
    for cp in comm_probs:
        print("Evaluating comm prob:", cp, "-> block_comm =", cp != -1.0)
        returns = np.array([])
        cfg.seed = 0
        with torch.no_grad():
            returns = np.concatenate((returns, run_eval(cfg, cp, cfg.n_eval_runs)))
        success = returns >= (cfg.episode_length * -1 + 60)

        results["Comm prob"].append(cp)
        results["Mean return"].append(returns.mean())
        results["Std"].append(returns.std())
        results["Median return"].append(np.median(returns))
        results["Success rate"].append(success.mean())
        results["Success std"].append(success.std())
        
        df = pd.DataFrame(results)
        df.to_csv(log_file_path, mode='w')
    
