import argparse
from matplotlib.pyplot import step
import torch
import sys
import imp
import os
import numpy as np
import pandas as pd
from gym.spaces import Box
from tensorboardX import SummaryWriter

from utils.buffer import ReplayBuffer
from utils.make_env import get_paths, load_scenario_config
from model.modules.maddpg import MADDPG

USE_CUDA = torch.cuda.is_available()

import time
from multiprocessing import Process, Pipe
from baselines.common.vec_env import VecEnv, CloudpickleWrapper


def worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x()
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            ob, reward, done, info = env.step(data)
            if all(done):
                ob = env.reset()
            remote.send((ob, reward, done, info))
        elif cmd == 'reset':
            ob = env.reset()
            remote.send(ob)
        elif cmd == 'reset_task':
            ob = env.reset_task()
            remote.send(ob)
        elif cmd == 'close':
            remote.close()
            break
        elif cmd == 'get_spaces':
            remote.send((env.observation_space, env.action_space))
        elif cmd == 'get_agent_types':
            if all([hasattr(a, 'adversary') for a in env.agents]):
                remote.send(['adversary' if a.adversary else 'agent' for a in
                             env.agents])
            else:
                remote.send(['agent' for _ in env.agents])
        else:
            raise NotImplementedError


class SubprocVecEnv(VecEnv):
    def __init__(self, env_fns, spaces=None):
        """
        envs: list of gym environments to run in subprocesses
        """
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
            for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = True # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(('get_spaces', None))
        observation_space, action_space = self.remotes[0].recv()
        self.remotes[0].send(('get_agent_types', None))
        self.agent_types = self.remotes[0].recv()
        VecEnv.__init__(self, len(env_fns), observation_space, action_space)

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def reset_task(self):
        for remote in self.remotes:
            remote.send(('reset_task', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:            
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True


class DummyVecEnv(VecEnv):
    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]        
        VecEnv.__init__(self, len(env_fns), env.observation_space, env.action_space)
        if all([hasattr(a, 'adversary') for a in env.agents]):
            self.agent_types = ['adversary' if a.adversary else 'agent' for a in
                                env.agents]
        else:
            self.agent_types = ['agent' for _ in env.agents]
        self.ts = np.zeros(len(self.envs), dtype='int')        
        self.actions = None

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        results = [env.step(a) for (a,env) in zip(self.actions, self.envs)]
        obs, rews, dones, infos = map(np.array, zip(*results))
        self.ts += 1
        for (i, done) in enumerate(dones):
            if all(done): 
                obs[i] = self.envs[i].reset()
                self.ts[i] = 0
        self.actions = None
        return np.array(obs), np.array(rews), np.array(dones), infos

    def reset(self):        
        results = [env.reset() for env in self.envs]
        return np.array(results)

    def close(self):
        return

    def render(self):
        self.envs[0].render()

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
            np.random.seed(seed + rank * 1000)
            return env
        return init_env
    if n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(n_rollout_threads)])

class ProgressBar:

    def __init__(self, max_number):
        self.max_number = max_number
        self.last_time = None
        self.last_number = None
        self.started_time = None
    
    def print_progress(self, current_number):
        # Compute duration of last iteration
        time_left = '?'
        current_time = time.time()
        if current_number > 0:
            time_since_start = current_time - self.started_time
            sec_per_i = time_since_start / current_number
            sec_left = (self.max_number - current_number) * sec_per_i
            time_left = time.strftime("%Hh%Mmin%Ss", time.gmtime(sec_left))
        else:
            self.started_time = current_time

        elapsed_time = time.strftime(
            "%Hh%Mmin%Ss", time.gmtime(current_time - self.started_time))

        percentage_done = 100 * current_number / self.max_number
        
        print(f"Step {current_number}/{self.max_number} ({percentage_done:0.2f}%), for {elapsed_time}, left {time_left}.", 
                end='\r')

        self.last_time = current_time
        self.last_number = current_number

    def print_end(self):
        elapsed_time = time.strftime(
            "%Hh%Mmin%Ss", time.gmtime(time.time() - self.started_time))
        print()
        print(f"Training ended after {elapsed_time}.")


def train_episode(env, model, replay_buffer, max_episode_length):
    ep_returns = 0.0
    ep_length = max_episode_length
    ep_success = False
    # Reset environment with initial positions
    obs = env.reset()
    for step_i in range(max_episode_length):
        # Perform step
        obs = np.array(obs)
        torch_obs = torch.Tensor(obs)
        actions = model.step(torch_obs, explore=True)
        actions = [a.squeeze().data.numpy() for a in actions]
        next_obs, rewards, dones, _ = env.step(actions)
        # Store experience in replay buffer
        replay_buffer.push(
            np.array([obs]), 
            np.array([np.expand_dims(a, axis=0) for a in actions]), 
            np.array([rewards]), 
            np.array([next_obs]), 
            np.array([dones]))
        # # rearrange observations to be per agent
        # torch_obs = [torch.Tensor(np.vstack(obs[:, a]))
        #              for a in range(model.n_agents)]
        # # get actions as torch Variables
        # torch_agent_actions = model.step(torch_obs, explore=True)
        # # convert actions to numpy arrays
        # agent_actions = [ac.data.numpy() for ac in torch_agent_actions]
        # # convert actions to numpy arrays
        # actions = [
        #     [ac[0] for ac in agent_actions]
        # ]
        # # Environment step
        # next_obs, rewards, dones, infos = env.step(actions)
        # replay_buffer.push(obs, agent_actions, rewards, next_obs, dones)

        ep_returns += rewards[0]
        if any(dones): # dones[0].any():
            ep_length = step_i + 1
            ep_success = True
            break
        obs = next_obs

    return ep_returns, ep_length, ep_success


def run(config):
    # Get paths for saving logs and model
    run_dir, model_cp_path, log_dir = get_paths(config)
    print("Saving model in dir", run_dir)

    # Save args in txt file
    with open(os.path.join(run_dir, 'args.txt'), 'w') as f:
        f.write(str(sys.argv))

    # Init summary writer
    logger = SummaryWriter(str(log_dir))

    # Load scenario config
    sce_conf = load_scenario_config(config, run_dir)

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # Set training device
    if USE_CUDA:
        if config.cuda_device is None:
            training_device = 'cuda'
        else:
            training_device = torch.device(config.cuda_device)
    else:
        training_device = 'cpu'
        torch.set_num_threads(config.n_training_threads)

    # env = make_parallel_env(config.env_path, 1, config.seed, 
    #                         config.discrete_action, sce_conf)
    env = make_env(config.env_path, sce_conf, config.discrete_action)

    maddpg = MADDPG(
        2, 17, 5, config.lr, config.gamma, config.tau, config.hidden_dim, 
        config.discrete_action, config.shared_params, config.init_exploration
    )

    replay_buffer = ReplayBuffer(
        config.buffer_length, 
        maddpg.n_agents,
        [obsp.shape[0] for obsp in env.observation_space],
        [acsp.shape[0] if isinstance(acsp, Box) else acsp.n
        for acsp in env.action_space]
    )

    # Number of explorations frames
    if config.n_exploration_frames is None:
        config.n_exploration_frames = config.n_frames

    # Initialise progress bar
    pb = ProgressBar(config.n_frames)
    
    print(f"Starting training for {config.n_frames} frames")
    print(f"                  updates every {config.frames_per_update} frames")
    print(f"                  with seed {config.seed}")
    train_data_dict = {
        "Step": [],
        "Episode return": [],
        "Success": [],
        "Episode length": []
    }
    step_i = 0
    last_train_i = 0
    last_save_i = 0
    while step_i < config.n_frames:
        # Print progess
        pb.print_progress(step_i)

        # Preparate model for rollout
        maddpg.prep_rollouts(device='cpu')

        # Set exploration noise
        explr_pct_remaining = max(0, config.n_exploration_frames - step_i) / \
            config.n_exploration_frames
        maddpg.scale_noise(config.final_exploration + \
            (config.init_exploration - config.final_exploration) * \
            explr_pct_remaining)
        maddpg.reset_noise()

        # Rollout episode
        ep_returns, ep_length, ep_success = train_episode(
            env, maddpg, replay_buffer, config.episode_length)
        step_i += ep_length
        
        # Training
        if (len(replay_buffer) >= config.batch_size and
           (step_i - last_train_i) >= config.frames_per_update):
            last_train_i = step_i
            maddpg.prep_training(device=training_device)
            for _ in range(config.n_training_per_updates):
                for a_i in range(maddpg.n_agents):
                    sample = replay_buffer.sample(config.batch_size,
                                                  cuda_device=training_device)
                    maddpg.update(sample, a_i)#, logger=logger)
            maddpg.update_all_targets()

        # Log
        # Log in list
        train_data_dict["Step"].append(step_i)
        train_data_dict["Episode return"].append(np.mean(ep_returns))
        train_data_dict["Success"].append(int(ep_success))
        train_data_dict["Episode length"].append(ep_length)
        # Tensorboard
        logger.add_scalar(
            'agent0/episode_return', 
            train_data_dict["Episode return"][-1], 
            train_data_dict["Step"][-1])

        # Save model
        if (step_i - last_save_i) >= config.save_interval:
            last_save_i = step_i
            os.makedirs(run_dir / 'incremental', exist_ok=True)
            maddpg.save(run_dir / 'incremental' / ('model_ep%i.pt' % (step_i)))
            maddpg.save(model_cp_path)
    
    pb.print_end()

    maddpg.save(model_cp_path)
    env.close()
    # Log Tensorboard
    logger.export_scalars_to_json(str(log_dir / 'summary.json'))
    logger.close()
    # Log csv
    rewards_df = pd.DataFrame(train_data_dict)
    rewards_df.to_csv(str(run_dir / 'mean_episode_rewards.csv'))
    print("Model saved in dir", run_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("env_path", help="Path to the environment")
    parser.add_argument("model_name",
                        help="Name of directory to store " +
                             "model/training contents")
    parser.add_argument("--seed", default=1, type=int, help="Random seed")
    # Environment
    parser.add_argument("--episode_length", default=100, type=int)
    parser.add_argument("--discrete_action", action='store_true')
    parser.add_argument("--sce_conf_path", default=None, type=str,
                        help="Path to the scenario config file")
    # Training
    parser.add_argument("--n_training_threads", default=6, type=int)
    parser.add_argument("--n_frames", default=25000, type=int)
    parser.add_argument("--buffer_length", default=int(1e6), type=int)
    parser.add_argument("--frames_per_update", default=100, type=int)
    parser.add_argument("--batch_size", default=512, type=int,
                        help="Batch size for model training")
    parser.add_argument("--n_training_per_updates", default=1, type=int)
    parser.add_argument("--n_exploration_frames", default=None, type=int)
    parser.add_argument("--init_exploration", default=0.6, type=float)
    parser.add_argument("--final_exploration", default=0.0, type=float)
    parser.add_argument("--save_interval", default=100000, type=int)
    # Model hyperparameters
    parser.add_argument("--hidden_dim", default=64, type=int)
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--tau", default=0.01, type=float)
    parser.add_argument("--gamma", default=0.95, type=float)
    parser.add_argument("--adversary_alg", default="MADDPG", type=str,
                        choices=['MADDPG', 'DDPG'])
    parser.add_argument("--shared_params", action='store_true')
    # Cuda
    parser.add_argument("--cuda_device", default=None, type=str)

    config = parser.parse_args()

    run(config)