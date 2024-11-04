import gym
import numpy as np

from itertools import chain

from .env_wrappers import DummyVecEnv, SubprocVecEnv
from .mpe.environment import MultiAgentEnv


def _get_env(cfg, init_pos):
    if "magym_PredPrey" in cfg.env_name:
        if "Respawn" in cfg.env_name:
            from .magym_PredPrey_Respawn.env import PredatorPreyEnv
        else:
            from .magym_PredPrey.env import PredatorPreyEnv
        env = PredatorPreyEnv(
            n_agents=cfg.magym_n_agents, 
            grid_shape=(cfg.magym_env_size, cfg.magym_env_size),
            n_preys=cfg.magym_n_preys, 
            max_steps=cfg.episode_length,
            agent_view_mask=(cfg.magym_obs_range, cfg.magym_obs_range),
            actual_obsrange=cfg.FT_magym_actual_obsrange,
            see_agents=cfg.magym_see_agents,
            init_pos=init_pos)
    elif cfg.env_name == "magym_Lumber":
        from .ma_gym.lumberjack import Lumberjacks
        env = Lumberjacks(
            n_agents=cfg.magym_n_agents, 
            grid_shape=(cfg.magym_env_size, cfg.magym_env_size), 
            max_steps=cfg.episode_length,
            agent_view_mask=(cfg.magym_obs_range, cfg.magym_obs_range),
            actual_obsrange=cfg.FT_magym_actual_obsrange)
    elif cfg.env_name == "magym_Foraging":
        from .magym_Foraging.env import Env
        env = Env(
            n_agents=cfg.magym_n_agents, 
            grid_shape=(cfg.magym_env_size, cfg.magym_env_size), 
            max_steps=cfg.episode_length,
            agent_view_mask=(cfg.magym_obs_range, cfg.magym_obs_range),
            no_purple=cfg.magym_no_purple,
            actual_obsrange=cfg.FT_magym_actual_obsrange)
    elif cfg.env_name == "magym_Foraging_fixedpos":
        from .magym_Foraging_fixedpos.env import Env
        env = Env(
            n_agents=cfg.magym_n_agents, 
            grid_shape=(cfg.magym_env_size, cfg.magym_env_size), 
            max_steps=cfg.episode_length,
            agent_view_mask=(cfg.magym_obs_range, cfg.magym_obs_range),
            no_purple=cfg.magym_no_purple,
            actual_obsrange=cfg.FT_magym_actual_obsrange)
    elif cfg.env_name == "magym_Combat":
        env = Env(
            grid_shape=(cfg.magym_env_size, cfg.magym_env_size), 
            max_steps=cfg.episode_length)
    elif cfg.env_name == "magym_Empty":
        from .magym_empty.env import EmptyEnv
        env = EmptyEnv(
            grid_shape=(cfg.magym_env_size, cfg.magym_env_size),
            n_agents=cfg.magym_n_agents, 
            max_steps=cfg.episode_length,
            agent_view_mask=(cfg.magym_obs_range, cfg.magym_obs_range),
            see_agents=cfg.magym_see_agents)
    return env

def _get_parser(cfg):
    if "magym_PredPrey" in cfg.env_name or cfg.env_name == "magym_Empty":
        if "Respawn" in cfg.env_name:
            from .magym_PredPrey_Respawn.parser import Parser
        else:
            from .magym_PredPrey.parser import Parser
        return Parser(cfg.magym_env_size, cfg.magym_obs_range, cfg.magym_n_preys)
    elif cfg.env_name == "magym_Lumber":
        from .parsers.lumberjack import Lumberjack_Parser
        return Lumberjack_Parser(cfg.magym_env_size)
    elif cfg.env_name == "magym_Foraging":
        from .magym_Foraging.parser import Parser
        return Parser(cfg.magym_env_size, cfg.magym_obs_range)
    elif cfg.env_name == "magym_Foraging_fixedpos":
        from .magym_Foraging_fixedpos.parser import Parser
        return Parser(cfg.magym_env_size, cfg.magym_obs_range)
    else:
        print("WARNING: No Parser for", cfg.env_name)
        return None

def reset_envs(envs):
    obs = envs.reset()
    share_obs = []
    for o in obs:
        share_obs.append(list(chain(*o)))
    share_obs = np.array(share_obs)
    return obs, share_obs

def make_env(cfg, n_threads, seed=None, init_positions=None):
    if seed is None:
        seed = cfg.seed

    def get_env_fn(rank, init_pos):
        def init_env():
            env = _get_env(cfg, init_pos)
            env.seed(seed + rank * 1000)
            return env
        return init_env

    parser = _get_parser(cfg)

    # assert init_positions is None or n_threads <= len(init_positions)

    if n_threads == 1:
        return DummyVecEnv([get_env_fn(0, init_positions[0] 
                            if init_positions is not None else None)]), \
                parser
    else:
        return SubprocVecEnv([get_env_fn(i, init_positions[i % len(init_positions)] 
                                    if init_positions is not None else None) 
                              for i in range(n_threads)]), \
                parser