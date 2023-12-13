import os
import time

from src.utils.config import get_config
from src.utils.utils import set_seeds, load_args, set_cuda_device
from src.envs.make_env import make_env
from src.lmc.lmc_context import LMC


def run():
     # Load config
    parser = get_config()
    cfg = parser.parse_args()

    # Get pretrained stuff
    assert cfg.model_dir is not None, "Must provide model_dir"
    load_args(cfg)
    pretrained_model_path = os.path.join(cfg.model_dir, "model_ep.pt")
    assert os.path.isfile(pretrained_model_path), "No model checkpoint provided."
    cfg.n_parallel_envs = 1

    set_seeds(cfg.seed)

    # Set training device
    device = set_cuda_device(cfg)
    
    # Create train environment
    envs, parser = make_env(cfg, cfg.n_parallel_envs)
    
    # Create model
    n_agents = envs.n_agents
    obs_space = envs.observation_space
    shared_obs_space = envs.shared_observation_space
    act_space = envs.action_space
    global_state_dim = envs.global_state_dim
    model = LMC(
        cfg, 
        n_agents, 
        obs_space, 
        shared_obs_space, 
        act_space,
        parser.vocab, 
        global_state_dim, 
        device)

    # Load params
    model.load(pretrained_model_path)

    obs, states = envs.reset()
    model.reset_context()
    model.prep_rollout(device)
    for ep_s_i in range(cfg.episode_length):
        # Parse obs
        parsed_obs = parser.get_perfect_messages(obs)
        # Perform step
        # Get action
        actions, broadcasts, agent_messages = model.comm_n_act(
            obs, parsed_obs)
        # Perform action and get reward and next obs
        obs, next_states, rewards, dones, infos = envs.step(actions)

        # Reward communication
        comm_rewards = model.eval_comm(rewards, agent_messages, states, dones)
        states = next_states

        envs.render("human")
        
        print(f"\nStep #{ep_s_i}")
        print("Messages", agent_messages)
        print("Perfect Messages", parsed_obs)
        print("Communication Rewards", comm_rewards)

        time.sleep(0.1)

        env_dones = dones.all(axis=1)
        if True in env_dones:
            break
            
    envs.close()

if __name__ == '__main__':
    run()