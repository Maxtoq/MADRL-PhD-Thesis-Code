import json
import imp
import sys
import os
import re
import numpy as np
from pathlib import Path
from shutil import copyfile
import wandb
import torch
from offpolicy.config import get_config
from offpolicy.utils.util import get_cent_act_dim, get_dim_from_space
from offpolicy.envs.mpe.environment import MultiAgentEnv
from offpolicy.envs.env_wrappers import DummyVecEnv, SubprocVecEnv


def get_paths(config):
    # Get environment name from script path
    env_name = re.findall("\/?([^\/.]*)\.py", config.scenario_name)[0]
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
            print('Special config for scenario:', config.scenario_name)
            print(sce_conf, '\n')
    return sce_conf

def make_env(scenario_name, sce_conf={}, discrete_action=False):
    '''
    Creates a MultiAgentEnv object as env. This can be used similar to a gym
    environment by calling env.reset() and env.step().
    Use env.render() to view the environment on the screen.

    Input:
        scenario_name   :   path of the scenario script
                            (without the .py extension)
        benchmark       :   whether you want to produce benchmarking data
                            (usually only done during evaluation)

    Some useful env properties (see environment.py):
        .observation_space  :   Returns the observation space for each agent
        .action_space       :   Returns the action space for each agent
        .n                  :   Returns the number of Agents
    '''

    # load scenario from script
    scenario = imp.load_source('', scenario_name).Scenario()
    # create world
    world = scenario.make_world(**sce_conf)
    # create multiagent environment
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, 
                        scenario.observation, 
                        done_callback=scenario.done if hasattr(scenario, "done") 
                        else None)
    return env

def make_train_env(all_args, sce_conf):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "MPE":
                env = make_env(all_args.scenario_name, sce_conf)
            else:
                print("Can not support the " +
                      all_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(all_args.seed + rank * 1000)
            return env
        return init_env
    if all_args.n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])


def make_eval_env(all_args, sce_conf):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "MPE":
                env = make_env(all_args.scenario_name, sce_conf)
            else:
                print("Can not support the " +
                      all_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(all_args.seed * 50000 + rank * 10000)
            return env
        return init_env
    if all_args.n_eval_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)])


def parse_args(args, parser):
    parser.add_argument('--scenario_name', type=str,
                        help="Path to scenario to train on")
    parser.add_argument("--num_landmarks", type=int, default=3)
    parser.add_argument('--num_agents', type=int,
                        default=3, help="number of agents")
    parser.add_argument('--use_same_share_obs', action='store_false',
                        default=True, help="Whether to use available actions")
    parser.add_argument("--sce_conf_path", default=None, type=str,
                        help="Path to the scenario config file")
    parser.add_argument("--model_name", type=str,
                        help="Name of directory to store " +
                             "models/training contents")

    all_args = parser.parse_known_args(args)[0]

    return all_args


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)

    # cuda and # threads
    if all_args.cuda and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    # Get paths for saving logs and model
    run_dir, _, _ = get_paths(all_args)

    # Load scenario config
    sce_conf = load_scenario_config(all_args, run_dir)

    # set seeds
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    # create env
    env = make_train_env(all_args, sce_conf)
    num_agents = sce_conf['nb_agents']

    # create policies and mapping fn
    if all_args.share_policy:
        policy_info = {
            'policy_0': {"cent_obs_dim": get_dim_from_space(env.share_observation_space[0]),
                         "cent_act_dim": get_cent_act_dim(env.action_space),
                         "obs_space": env.observation_space[0],
                         "share_obs_space": env.share_observation_space[0],
                         "act_space": env.action_space[0]}
        }

        def policy_mapping_fn(id): return 'policy_0'
    else:
        policy_info = {
            'policy_' + str(agent_id): {"cent_obs_dim": get_dim_from_space(env.share_observation_space[agent_id]),
                                        "cent_act_dim": get_cent_act_dim(env.action_space),
                                        "obs_space": env.observation_space[agent_id],
                                        "share_obs_space": env.share_observation_space[agent_id],
                                        "act_space": env.action_space[agent_id]}
            for agent_id in range(num_agents)
        }

        def policy_mapping_fn(agent_id): return 'policy_' + str(agent_id)

    # choose algo
    if all_args.algorithm_name in ["rmatd3", "rmaddpg", "rmasac", "qmix", "vdn"]:
        from offpolicy.runner.rnn.mpe_runner import MPERunner as Runner
        assert all_args.n_rollout_threads == 1, (
            "only support 1 env in recurrent version.")
        eval_env = env
    elif all_args.algorithm_name in ["matd3", "maddpg", "masac", "mqmix", "mvdn"]:
        from offpolicy.runner.mlp.mpe_runner import MPERunner as Runner
        eval_env = make_eval_env(all_args, sce_conf)
    else:
        raise NotImplementedError

    config = {"args": all_args,
              "policy_info": policy_info,
              "policy_mapping_fn": policy_mapping_fn,
              "env": env,
              "eval_env": eval_env,
              "num_agents": num_agents,
              "device": device,
              "use_same_share_obs": all_args.use_same_share_obs,
              "run_dir": run_dir
              }

    total_num_steps = 0
    runner = Runner(config=config)
    while total_num_steps < all_args.num_env_steps:
        total_num_steps = runner.run()

    env.close()
    if all_args.use_eval and (eval_env is not env):
        eval_env.close()

    runner.writter.export_scalars_to_json(
            str(runner.log_dir + '/summary.json'))
    runner.writter.close()


if __name__ == "__main__":
    main(sys.argv[1:])
