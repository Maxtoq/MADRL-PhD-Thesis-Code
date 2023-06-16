import gym

from .env_wrappers import DummyVecEnv, SubprocVecEnv

ma_gym_tasks = {
    "Switch2": "ma_gym:Switch2-v0"
}


def make_env(cfg, n_rollout_threads):
    def get_env_fn(rank):
        def init_env():
            if cfg.env_name == "ma_gym":
                env = gym.make(ma_gym_tasks[cfg.task_name])
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