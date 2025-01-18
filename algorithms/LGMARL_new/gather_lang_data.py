import numpy as np
import pandas as pd

from tqdm import trange

from src.utils.config import get_config
from src.envs.make_env import make_env


def run():
     # Load config
    argparse = get_config()
    cfg = argparse.parse_args()
    
    # Create train environment
    envs, parser = make_env(cfg, cfg.n_parallel_envs)

    data = {
        "obs": [],
        "lang": []
    }

    obs = envs.reset()
    parsed_obs = parser.get_perfect_messages(obs)

    for s_i in trange(0, cfg.n_steps, cfg.n_parallel_envs, ncols=0):
        for e_i in range(cfg.n_parallel_envs):
            for a_i in range(cfg.magym_n_agents):
                data["obs"].append(list(obs[e_i, a_i]))
                data["lang"].append(' '.join(parsed_obs[e_i][a_i]))

        actions = np.random.randint(0, 5, (cfg.n_parallel_envs, cfg.magym_n_agents, 1))

        # Perform action and get reward and next obs
        obs, rewards, dones, infos = envs.step(actions)

        parsed_obs = parser.get_perfect_messages(obs)
            
    envs.close()
    
    df = pd.DataFrame(data)
    df.to_csv("results/data/lamarl_data/" + cfg.experiment_name + ".csv")

if __name__ == '__main__':
    run()
