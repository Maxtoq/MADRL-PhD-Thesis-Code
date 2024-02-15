import os
import time
import numpy as np

from src.utils.config import get_config
from src.utils.utils import set_seeds, load_args, set_cuda_device
from src.envs.make_env import make_env
# from src.lmc.lmc_context import LMC


def run():
    # Load config
    parser = get_config()
    cfg = parser.parse_args()

    set_seeds(cfg.seed)

    # Set training device
    device = set_cuda_device(cfg)
    
    # Create train environment
    envs, parser = make_env(cfg, cfg.n_parallel_envs)

    obs = envs.reset()
    if cfg.use_render:
        envs.render("human")
    if cfg.render_wait_input:
        input()
    else:
        time.sleep(0.1)

    for ep_s_i in range(cfg.episode_length):
        # Parse obs
        parsed_obs = parser.get_perfect_messages(obs)
        # Perform step
        # Get action
        actions = np.random.randint(0, 5, (1, 4, 1))

        # Perform action and get reward and next obs
        obs, rewards, dones, infos = envs.step(actions)
        
        print(f"\nStep #{ep_s_i}")
        print("Observations", obs)
        print("Perfect Messages", parsed_obs)
        print("Actions", actions)
        print("Rewards", rewards)
        # print("Messages", agent_messages)

        if cfg.use_render:
            envs.render("human")
        if cfg.render_wait_input:
            input()
        else:
            time.sleep(0.1)

        # Reward communication
        # comm_rewards = model.eval_comm(rewards, agent_messages, states, dones)
        # states = next_states

        # print("Communication Rewards", comm_rewards)

        env_dones = dones.all(axis=1)
        if True in env_dones:
            print("ENV DONE")
            break
            
    envs.close()

if __name__ == '__main__':
    run()