import os
import time
import numpy as np

from tqdm import trange

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

def run():
    # Load config
    parser = get_config()
    cfg = parser.parse_args()

    # Get pretrained stuff
    assert cfg.model_dir is not None, "Must provide model_dir"
    load_args(cfg, eval=True)
    pretrained_model_path = os.path.join(cfg.model_dir, "model_ep.pt")
    assert os.path.isfile(pretrained_model_path), "No model checkpoint provided."
    print("Starting eval with config:")
    print(cfg)
    cfg.comm_type = "language"

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
    model = LanguageGroundedMARL(
        cfg, 
        n_agents, 
        obs_space, 
        shared_obs_space, 
        act_space,
        parser, 
        device,
        comm_eps_start=0.0)

    # Load params
    model.load(pretrained_model_path)
    model.set_eval()

    obs = envs.reset()
    parsed_obs = parser.get_perfect_messages(obs)

    render(cfg, envs)

    model.init_episode(obs, parsed_obs)
    model.prep_rollout(device)

    count_returns = np.zeros(cfg.n_parallel_envs)
    returns = []
    for ep_s_i in trange(0, cfg.n_steps, cfg.n_parallel_envs):
        add_mess = interact(cfg)
        
        # Get action
        actions, broadcasts, agent_messages, comm_rewards = model.comm_n_act(
            add_mess)

        # Perform action and get reward and next obs
        next_obs, rewards, dones, infos = envs.step(actions)

        decoded_messages = model.lang_learner.word_encoder.decode_batch(
            agent_messages[0])
        
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

        env_dones = dones.all(axis=1)
        if True in env_dones:
            print("ENV DONE")
            model.reset_context(env_dones)
            returns += list(count_returns[env_dones == True])
            count_returns *= (1 - env_dones)
            
    envs.close()

    print("Done", len(returns), "complete episodes.")
    print("Mean return:", sum(returns) / len(returns))
    input()

if __name__ == '__main__':
    run()