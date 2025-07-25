import gym
import numpy as np

from itertools import chain

from .env_wrappers import DummyVecEnv, SubprocVecEnv
from .mpe.environment import MultiAgentEnv


def _get_env(cfg, init_pos):
    if cfg.env_name == "magym_PredPrey_RGB":
        from .magym_PredPrey_RGB.env import PredatorPreyEnv
        env = PredatorPreyEnv(
            n_agents=cfg.n_agents, 
            grid_shape=(cfg.magym_env_size, cfg.magym_env_size),
            n_preys=cfg.n_preys, 
            max_steps=cfg.episode_length,
            obs_range=cfg.magym_obs_range,
            reduced_obsrange=cfg.FT_magym_reduced_obsrange,
            see_agents=cfg.magym_see_agents,
            init_pos=init_pos)

    elif "magym_PredPrey" in cfg.env_name:
        if "Respawn" in cfg.env_name:
            from .magym_PredPrey_Respawn.env import PredatorPreyEnv
        else:
            from .magym_PredPrey.env import PredatorPreyEnv
        env = PredatorPreyEnv(
            n_agents=cfg.n_agents, 
            grid_shape=(cfg.magym_env_size, cfg.magym_env_size),
            n_preys=cfg.n_preys, 
            max_steps=cfg.episode_length,
            agent_view_mask=(cfg.magym_obs_range, cfg.magym_obs_range),
            actual_obsrange=cfg.FT_magym_reduced_obsrange,
            see_agents=cfg.magym_see_agents,
            init_pos=init_pos)

    # elif cfg.env_name == "magym_Lumber":
    #     from .ma_gym.lumberjack import Lumberjacks
    #     env = Lumberjacks(
    #         n_agents=cfg.n_agents, 
    #         grid_shape=(cfg.magym_env_size, cfg.magym_env_size), 
    #         max_steps=cfg.episode_length,
    #         agent_view_mask=(cfg.magym_obs_range, cfg.magym_obs_range),
    #         actual_obsrange=cfg.FT_magym_reduced_obsrange)

    elif "magym_Foraging" in cfg.env_name:
        if cfg.env_name == "magym_Foraging_fixedpos":
            from .magym_Foraging_fixedpos.env import Env
        elif cfg.env_name == "magym_Foraging_RGB":
            from .magym_Foraging_RGB.env import Env
        elif cfg.env_name == "magym_ForagingHostile_RGB":
            from .magym_ForagingHostile_RGB.env import Env
        else:
            from .magym_Foraging.env import Env
        env = Env(
            n_agents=cfg.n_agents, 
            grid_shape=(cfg.magym_env_size, cfg.magym_env_size), 
            max_steps=cfg.episode_length,
            obs_range=cfg.magym_obs_range,
            reduced_obsrange=cfg.FT_magym_reduced_obsrange)

    # elif cfg.env_name == "magym_Combat":
    #     env = Env(
    #         grid_shape=(cfg.magym_env_size, cfg.magym_env_size), 
    #         max_steps=cfg.episode_length)

    elif cfg.env_name == "magym_CoordPlace_RGB":
        from .magym_CoordPlace_RGB.env import Env
        env = Env(
            n_agents=cfg.n_agents, 
            max_steps=cfg.episode_length,
            stop_done=True)

    elif cfg.env_name == "magym_Empty":
        from .magym_empty.env import EmptyEnv
        env = EmptyEnv(
            grid_shape=(cfg.magym_env_size, cfg.magym_env_size),
            n_agents=cfg.n_agents, 
            max_steps=cfg.episode_length,
            agent_view_mask=(cfg.magym_obs_range, cfg.magym_obs_range),
            see_agents=cfg.magym_see_agents)

    elif cfg.env_name == "magym_Empty_RGB":
        from .magym_empty_RGB.env import EmptyEnv
        env = EmptyEnv(
            grid_shape=(cfg.magym_env_size, cfg.magym_env_size),
            n_agents=cfg.n_agents, 
            max_steps=cfg.episode_length,
            obs_range=cfg.magym_obs_range,
            see_agents=cfg.magym_see_agents)
        
    elif cfg.env_name == "mpe_PredPrey":
        from .mpe_PredPrey.env import Scenario
        scenario = Scenario()
        scenario.make_world(
            cfg.n_agents, cfg.n_preys, max_steps=cfg.episode_length)
        env = MultiAgentEnv(scenario, discrete_action=cfg.mpe_discrete_action)
        
    elif cfg.env_name == "mpe_PredPrey_shape":
        from .mpe_PredPrey_shape.env import Scenario
        scenario = Scenario()
        scenario.make_world(
            cfg.n_agents, cfg.n_preys, max_steps=cfg.episode_length)
        env = MultiAgentEnv(scenario, discrete_action=cfg.mpe_discrete_action)

    elif cfg.env_name == "mpe_simple_color_reference":
        from .mpe_simple_color_reference.env import Scenario
        scenario = Scenario()
        scenario.make_world(cfg.episode_length)
        env = MultiAgentEnv(scenario, discrete_action=cfg.mpe_discrete_action)

    elif cfg.env_name == "mpe_simple_hardcolor_reference":
        from .mpe_simple_hardcolor_reference.env import Scenario
        scenario = Scenario()
        scenario.make_world(cfg.episode_length)
        env = MultiAgentEnv(scenario, discrete_action=cfg.mpe_discrete_action)

    else:
        raise NotImplementedError("ARG ERROR: bad env_name")
    return env

def _get_parser(cfg):
    if "magym_PredPrey" in cfg.env_name or cfg.env_name == "magym_Empty":
        if "Respawn" in cfg.env_name:
            from .magym_PredPrey_Respawn.parser import Parser
        elif cfg.env_name == "magym_PredPrey_RGB":
            from .magym_PredPrey_RGB.parser import Parser
        else:
            from .magym_PredPrey.parser import Parser
        return Parser(cfg.magym_env_size, cfg.magym_obs_range, cfg.n_preys)

    elif cfg.env_name == "magym_Lumber":
        from .parsers.lumberjack import Lumberjack_Parser
        return Lumberjack_Parser(cfg.magym_env_size)

    elif "magym_Foraging" in cfg.env_name:
        if cfg.env_name == "magym_Foraging_fixedpos":
            from .magym_Foraging_fixedpos.parser import Parser
        elif cfg.env_name == "magym_Foraging_RGB":
            from .magym_Foraging_RGB.parser import Parser
        elif cfg.env_name == "magym_ForagingHostile_RGB":
            from .magym_ForagingHostile_RGB.parser import Parser
        else:
            from .magym_Foraging.parser import Parser
        return Parser(cfg.magym_env_size, cfg.magym_obs_range)
    
    elif cfg.env_name == "magym_CoordPlace_RGB":
        from .magym_CoordPlace_RGB.parser import Parser
        return Parser()
    
    elif cfg.env_name == "magym_Empty_RGB":
        from .magym_empty_RGB.parser import Parser
        return Parser(cfg.magym_env_size, cfg.magym_obs_range)
        
    elif cfg.env_name == "mpe_PredPrey":
        from .mpe_PredPrey.parser import Parser
        return Parser(cfg.n_agents, cfg.n_preys)
        
    elif cfg.env_name == "mpe_PredPrey_shape":
        from .mpe_PredPrey_shape.parser import Parser
        return Parser(cfg.n_agents, cfg.n_preys)
    
    elif cfg.env_name == "mpe_simple_color_reference":
        from .mpe_simple_color_reference.parser import Parser
        return Parser()
    
    elif cfg.env_name == "mpe_simple_hardcolor_reference":
        from .mpe_simple_hardcolor_reference.parser import Parser
        return Parser()

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

def make_env(cfg, seed=None, init_positions=None):
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

    if cfg.n_parallel_envs == 1:
        return DummyVecEnv([get_env_fn(0, init_positions[0] 
                            if init_positions is not None else None)]), \
                parser
    else:
        return SubprocVecEnv([get_env_fn(i, init_positions[i % len(init_positions)] 
                                    if init_positions is not None else None) 
                              for i in range(cfg.n_parallel_envs)]), \
                parser
    
# def mod_envs(cfg, envs, init_positions=None):
#     def get_env_fn(rank, init_pos):
#         def init_env():
#             env = _get_env(cfg, init_pos)
#             env.seed(cfg.seed + rank * 1000)
#             return env
#         return init_env
    
#     parser = _get_parser(cfg)

#     if cfg.n_parallel_envs == 1:
#         envs.update_envs(
#             [get_env_fn(0, init_positions[0] 
#                 if init_positions is not None else None)])
#         return parser
#     else:
#         envs.update_envs(
#             [get_env_fn(i, init_positions[i % len(init_positions)] 
#                 if init_positions is not None else None) 
#              for i in range(cfg.n_parallel_envs)])
#         return parser