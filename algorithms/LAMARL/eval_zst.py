import os
import json
import time
import torch
import random
import itertools
import numpy as np
import pandas as pd

from tqdm import trange
from itertools import permutations

from src.utils.config import get_config
from src.utils.utils import set_seeds, set_cuda_device, load_args
from src.envs.make_env import make_env
from src.algo.lgmarl_diff import LanguageGroundedMARL




def get_unique_mappings(pattern):
    unique_letters = sorted(set(pattern))
    num_ids = 4  # IDs: 0, 1, 2, 3
    
    if len(unique_letters) > num_ids:
        return []  # Impossible to assign more unique agents than IDs
    
    # All permutations of IDs for unique letters
    return permutations(range(num_ids), len(pattern))

def generate_teams(pattern):
    unique_letters = sorted(set(pattern))
    letter_to_index = {letter: i for i, letter in enumerate(unique_letters)}
    index_pattern = [letter_to_index[char] for char in pattern]
    
    teams = set()
    for mapping in get_unique_mappings(pattern):
        team = tuple(mapping[i] for i in index_pattern)
        teams.update(permutations(team))
    
    return sorted(teams)

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

def get_team_compo(paths, team_compo):
    # ids = list(range(len(team_compo)))
    # random.shuffle(ids)
    team = [paths[team_compo[i]] for i in range(len(team_compo))]
    return team

def compute_ci(data, c=0.95):
    import scipy.stats as stats
    ci = stats.t.interval(
        c, 
        df=len(data)-1, 
        loc=np.mean(data), 
        scale=np.std(data, ddof=1) / np.sqrt(len(data)))
    return ci


def run_eval(cfg, team_compo, n_eval_runs):
    set_seeds(cfg.seed)

    # Get pretrained stuff
    assert cfg.model_dir is not None, "Must provide model_dir"
    load_args(cfg, eval=True)
    if ',' in cfg.model_dir:
        paths = cfg.model_dir.split(',')

        used_paths = get_team_compo(paths, team_compo)

        pretrained_model_path = []
        for p in used_paths:
            pretrained_model_path.append(os.path.join(p, "model_ep.pt"))
            assert os.path.isfile(pretrained_model_path[-1]), "No model checkpoint found at" + str(pretrained_model_path[-1])
        # print("Loading checkpoints from runs", used_paths)
    else:
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

    # Load params
    model.load(pretrained_model_path)
    model.prep_rollout(device)

    all_returns = np.array([])
    all_ep_lens = np.array([])
    for e_i in range(n_eval_runs):
        obs = envs.reset()
        parsed_obs = None # parser.get_perfect_messages(obs)
        model.init_episode(obs, parsed_obs)

        count_returns = np.zeros(cfg.n_parallel_envs)
        returns = np.zeros(cfg.n_parallel_envs)
        env_dones = np.zeros(cfg.n_parallel_envs)
        ep_lens = np.zeros(cfg.n_parallel_envs)
        for ep_s_i in range(0, cfg.episode_length):
            # Get action
            actions, agent_messages, _, comm_rewards \
                = model.act(deterministic=True)

            # Perform action and get reward and next obs
            next_obs, rewards, dones, infos = envs.step(actions)


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

            obs = next_obs
            parsed_obs = None # parser.get_perfect_messages(obs)
            model.store_exp(obs, parsed_obs, rewards, dones)

            count_returns += rewards.mean(-1)
            render(cfg, envs)

            dones = dones.all(axis=1)
            new_dones = dones * (1 - env_dones) == True
            if True in new_dones:
                env_dones += new_dones

                returns[new_dones] = count_returns[new_dones]
                ep_lens[new_dones] = ep_s_i + 1
                # count_returns *= (1 - env_dones)
                if ep_s_i == cfg.rollout_length - 1 or env_dones.all():
                    break

        all_returns = np.concatenate((all_returns, returns))
        all_ep_lens = np.concatenate((all_ep_lens, ep_lens))
            
    envs.close()

    return all_returns, all_ep_lens

if __name__ == '__main__':
    # Load config
    parser = get_config()
    cfg = parser.parse_args()
    cfg.seed = 0

    # team_compo = [
    #     "AAAA", "AAAB", "AABB", "AABC", "ABCD"]
    team_compo = ["AA", "AB"]
    results = {
        "Team compo": [],
        "Mean return": [],
        "Std return": [],
        "Median return": [],
        "Ci lower return": [],
        "Ci upper return": [],
        "Mean success rate": [],
        "Std success rate": [],
        "Median success rate": [],
        "Ci lower success rate": [],
        "Ci upper success rate": [],
    }
    log_file_path = '/'.join(cfg.model_dir.split(',')[0].split('/')[:-1]) + "/zst_log_E.csv"
    for tc in team_compo:
        print("Evaluating team composition:", tc)
        teams = generate_teams(tc)
        # returns = np.array([])
        # # seeds = []
        # for i in trange(24): # 24 = max number of permutations
        #     cfg.seed = 0
        #     with torch.no_grad():
        #         returns = np.concatenate((returns, run_eval(cfg, permuts[i % len(permuts)], cfg.n_eval_runs // 24)))
        #     success = returns >= (cfg.episode_length * -1 + 60)

        return_means = []
        success_rates = []
        for i in trange(len(teams)):
            with torch.no_grad():
                returns, ep_lens = run_eval(cfg, teams[i], cfg.n_eval_runs)

                success = ep_lens < cfg.episode_length

                return_means.append(returns.mean())
                success_rates.append(success.mean())

        return_means = np.array(return_means)
        success_rates = np.array(success_rates)

        # TODO: for each zst level, compute the mean/median/std/ci-upper-lower of returns/success
        ci_upper, ci_lower  = compute_ci(return_means)

        results["Team compo"].append(tc)
        results["Mean return"].append(return_means.mean())
        results["Std return"].append(return_means.std())
        results["Median return"].append(np.median(return_means))
        ci_upper, ci_lower  = compute_ci(return_means)
        results["Ci lower return"].append(ci_lower)
        results["Ci upper return"].append(ci_upper)
        results["Mean success rate"].append(success_rates.mean())
        results["Std success rate"].append(success_rates.std())
        results["Median success rate"].append(np.median(success_rates))
        ci_upper, ci_lower  = compute_ci(success_rates)
        results["Ci lower success rate"].append(ci_lower)
        results["Ci upper success rate"].append(ci_upper)
        
        df = pd.DataFrame(results)
        df.to_csv(log_file_path, mode='w')
    
