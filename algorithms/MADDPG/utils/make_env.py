import numpy as np
import json
import imp
import os
import re
from pathlib import Path
from shutil import copyfile
from .env_wrappers import SubprocVecEnv, DummyVecEnv

def make_env(scenario_path, sce_conf={}, discrete_action=False):
    '''
    Creates a MultiAgentEnv object as env. This can be used similar to a gym
    environment by calling env.reset() and env.step().
    Use env.render() to view the environment on the screen.

    Input:
        scenario_path   :   path of the scenario script
                            (without the .py extension)
        benchmark       :   whether you want to produce benchmarking data
                            (usually only done during evaluation)

    Some useful env properties (see environment.py):
        .observation_space  :   Returns the observation space for each agent
        .action_space       :   Returns the action space for each agent
        .n                  :   Returns the number of Agents
    '''
    from multiagent.environment import MultiAgentEnv

    # load scenario from script
    scenario = imp.load_source('', scenario_path).Scenario()
    # create world
    world = scenario.make_world(**sce_conf)
    # create multiagent environment
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward,
                        scenario.observation, 
                        done_callback=scenario.done if hasattr(scenario, "done")
                        else None, discrete_action=discrete_action)
    return env

def make_parallel_env(env_path, n_rollout_threads, seed, discrete_action, 
                      sce_conf={}):
    def get_env_fn(rank):
        def init_env():
            env = make_env(env_path, discrete_action=discrete_action, 
                           sce_conf=sce_conf)
            env.seed(seed + rank * 1000)
            np.random.seed(seed + rank * 1000)
            return env
        return init_env
    if n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(n_rollout_threads)])

def get_paths(config):
    # Get environment name from script path
    env_name = re.findall("\/?([^\/.]*)\.py", config.env_path)[0]
    # Get path of the run directory
    model_dir = Path('./models') / env_name / config.model_name
    if not model_dir.exists():
        curr_run = 'run1'
    else:
        exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in
                         model_dir.iterdir() if
                         str(folder.name).startswith('run')]
        if len(exst_run_nums) == 0:
            curr_run = 'run1'
        else:
            curr_run = 'run%i' % (max(exst_run_nums) + 1)
    run_dir = model_dir / curr_run
    model_cp_path = run_dir / 'model.pt'
    log_dir = run_dir / 'logs'
    if not log_dir.exists():
        os.makedirs(log_dir)

    return run_dir, model_cp_path, log_dir

def load_scenario_config(config, run_dir):
    sce_conf = {}
    if config.sce_conf_path is not None:
        copyfile(config.sce_conf_path, run_dir / 'sce_config.json')
        with open(config.sce_conf_path) as cf:
            sce_conf = json.load(cf)
            print('Special config for scenario:', config.env_path)
            print(sce_conf, '\n')
    return sce_conf