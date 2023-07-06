import gym
import numpy as np

from itertools import chain

from .env_wrappers import DummyVecEnv, SubprocVecEnv
from .mpe.environment import MultiAgentEnv

def _get_env(cfg):
    if "mpe" in cfg.env_name:
        if "Spread" in cfg.env_name:
            from .mpe.scenarios.simple_spread import Scenario
        if "CoopPush" in cfg.env_name:
            from .mpe.scenarios.coop_push_corners import Scenario
        scenario = Scenario()
        scenario.make_world()
        env = MultiAgentEnv(scenario, discrete_action=True)
    elif "magym" in cfg.env_name:
        if "Switch2" in cfg.env_name:
            from .ma_gym.envs.switch.switch_one_corridor import Switch
            env = Switch(n_agents=2, max_steps=cfg.episode_length, clock=False)
    elif "rel_overgen" in cfg.env_name:
        from .rel_overgen import RelOvergenEnv
        env = RelOvergenEnv(
            cfg.ro_state_dim, optim_diff_coeff=cfg.ro_optim_diff_coeff)
    return env

def reset_envs(envs):
    obs = envs.reset()
    share_obs = []
    for o in obs:
        share_obs.append(list(chain(*o)))
    share_obs = np.array(share_obs)
    return obs, share_obs

def make_env(cfg, n_rollout_threads, seed=None):
    if seed is None:
        seed = cfg.seed
    def get_env_fn(rank):
        def init_env():
            env = _get_env(cfg)
            # env.seed(seed + rank * 1000)
            return env
        return init_env
    if n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([
            get_env_fn(i) for i in range(n_rollout_threads)])