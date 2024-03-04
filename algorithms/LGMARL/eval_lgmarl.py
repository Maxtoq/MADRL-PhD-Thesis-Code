import os
import time
import numpy as np

from src.utils.config import get_config
from src.utils.utils import set_seeds, load_args, set_cuda_device
from src.envs.make_env import make_env
from src.algo.lgmarl import LanguageGroundedMARL

def render(cfg, envs):
    if cfg.use_render:
        envs.render("human")
    if cfg.render_wait_input:
        input()
    else:
        time.sleep(0.1)

def run():
    # Load config
    parser = get_config()
    cfg = parser.parse_args()

    # Get pretrained stuff
    assert cfg.model_dir is not None, "Must provide model_dir"
    load_args(cfg)
    pretrained_model_path = os.path.join(cfg.model_dir, "model_ep.pt")
    assert os.path.isfile(pretrained_model_path), "No model checkpoint provided."
    print("Starting eval with config:")
    print(cfg)
    # cfg.comm_type = "language"

    set_seeds(cfg.seed)

    # Set training device
    device = set_cuda_device(cfg)
    
    # Create train environment
    envs, parser = make_env(cfg, 1)
    
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
        device)

    # Load params
    model.load(pretrained_model_path)

    obs = envs.reset()
    parsed_obs = parser.get_perfect_messages(obs)

    render(cfg, envs)

    model.init_episode(obs, parsed_obs)
    model.prep_rollout(device)
    for ep_s_i in range(1000):#cfg.episode_length):
        # Get action
        actions, broadcasts, agent_messages, comm_rewards = model.comm_n_act(
            parsed_obs)

        # Perform action and get reward and next obs
        next_obs, rewards, dones, infos = envs.step(actions)

        obs = next_obs
        parsed_obs = parser.get_perfect_messages(obs)
        
        print(f"\nStep #{ep_s_i + 1}")
        print("Observations", obs)
        print("Perfect Messages", parsed_obs)
        print("Agent Messages", agent_messages)
        print("Actions (t-1)", actions)
        print("Rewards", rewards)
        print("Communication Rewards", comm_rewards)

        render(cfg, envs)

        env_dones = dones.all(axis=1)
        if True in env_dones:
            print("ENV DONE")
            model.init_episode()
            
    envs.close()

if __name__ == '__main__':
    run()