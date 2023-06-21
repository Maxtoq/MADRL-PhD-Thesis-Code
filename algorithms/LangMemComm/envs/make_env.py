import gym

from .env_wrappers import DummyVecEnv, SubprocVecEnv


def get_env_class_and_args(cfg):
    if cfg.task_name == "Switch2":
        from .ma_gym.envs.switch.switch_one_corridor import Switch as EnvClass
        args = {
            "n_agents": 2,
            "max_steps": cfg.episode_length,
            "clock": False
        }
        return EnvClass, args

def make_env(cfg, n_rollout_threads):
    env_class, args = get_env_class_and_args(cfg)
    def get_env_fn(rank):
        def init_env():
            if cfg.env_name == "ma_gym":
                env = env_class(**args)
            else:
                print("Can not support the " +
                      cfg.env_name + "environment.")
                raise NotImplementedError
            env.seed(cfg.seed + rank * 1000)
            return env
        return init_env
    if n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([
            get_env_fn(i) for i in range(n_rollout_threads)])