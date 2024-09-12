import os
import json
import time
import random
import numpy as np

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
    # print("Starting eval with config:")
    # print(cfg)
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

    obs = envs.reset()
    parsed_obs = parser.get_perfect_messages(obs)

    render(cfg, envs)

    model.init_episode(obs, parsed_obs)
    model.prep_rollout(device)

    count_returns = np.zeros(cfg.n_parallel_envs)
    # returns = []
    returns = np.zeros(cfg.n_parallel_envs)
    env_dones = np.zeros(cfg.n_parallel_envs)
    for ep_s_i in trange(0, cfg.rollout_length):
        # add_mess = interact(cfg)
        
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
        parsed_obs = parser.get_perfect_messages(obs)
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
            
    envs.close()

    print("Done", len(returns), "complete episodes.")
    print("Returns", returns)
    print("Mean return:", sum(returns) / len(returns))

    return sum(returns) / len(returns)

if __name__ == '__main__':
    # Load config
    parser = get_config()
    cfg = parser.parse_args()

    random.seed(cfg.seed)

    returns = np.zeros(cfg.n_eval_runs)
    for i in range(cfg.n_eval_runs):
        print(f"Run {i + 1}/{cfg.n_eval_runs}")
        cfg.seed = random.randint(0, 100000)
        returns[i] = run_eval(cfg)

    print(returns, returns.mean(), returns.std(), np.median(returns))
    