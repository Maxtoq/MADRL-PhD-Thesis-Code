import os
import time
import numpy as np

from src.utils.config import get_config
from src.utils.utils import set_seeds, load_args, set_cuda_device, render
from src.envs.make_env import make_env
# from src.lmc.lmc_context import LMC


# def render(envs):
#     envs.render("human")
#     # input()
#     time.sleep(0.1)

def run():
    # Load config
    parser = get_config()
    cfg = parser.parse_args()

    set_seeds(cfg.seed)

    # Set training device
    device = set_cuda_device(cfg)
    
    # Create train environment
    envs, parser = make_env(cfg)
    
    obs = envs.reset()
    # Parse obs
    if parser is not None:
        parsed_obs = parser.get_perfect_messages(obs)
    else:
        parsed_obs = None
    print("Observations", obs)
    print("Perfect Messages", parsed_obs)
    render(cfg, envs)

    for ep_s_i in range(cfg.episode_length):
        # Perform step
        # Get action
        actions = np.random.randint(0, 5, (1, cfg.magym_n_agents, 1))

        # Perform action and get reward and next obs
        obs, rewards, dones, infos = envs.step(actions)
        # Parse obs
        if parser is not None:
            parsed_obs = parser.get_perfect_messages(obs)
        else:
            parsed_obs = None
        
        print(f"\nStep #{ep_s_i + 1}")
        print("Observations", obs)
        print("Perfect Messages", parsed_obs)
        print("Actions (t-1)", actions)
        print("Rewards", rewards)

        render(cfg, envs)

        env_dones = dones.all(axis=1)
        if True in env_dones:
            print("ENV DONE")
            # break
            
    envs.close()

if __name__ == '__main__':
    run()