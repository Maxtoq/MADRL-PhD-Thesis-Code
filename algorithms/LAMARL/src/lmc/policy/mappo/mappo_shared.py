import torch
import numpy as np

from .buffer_shared import SharedReplayBuffer
from .policy import R_MAPPOPolicy
from .train_algo import R_MAPPOTrainAlgo

def torch2numpy(x):
    return x.detach().cpu().numpy()


class MAPPO:
    """
    Class handling training of MAPPO from paper "The Surprising Effectiveness 
    of PPO in Cooperative Multi-Agent Games" (https://arxiv.org/abs/2103.01955),
    with shared parameters accross all agents.
    :param args: (argparse.Namespace) all arguments for training
    :param n_agents: (int) number of agents
    :param obs_dim: (int) observation dimensions, for each agent
    :param shared_obs_dim: (int) centralised observation dimensions, for 
        each agent
    :param act_space: (gym.Space) action dimensions, for each agent
    :param device: (torch.device) cuda device used for training
    """
    def __init__(self, 
            args, n_agents, obs_dim, shared_obs_dim, act_space, device):
        self.args = args
        self.n_agents = n_agents
        self.n_parallel_envs = args["n_parallel_envs"]
        self.recurrent_N = args["recurrent_N"]
        self.use_centralized_V = self.args["use_centralized_V"]
        self.train_device = device

        # Set variant
        if self.args["policy_algo"] == "rmappo":
            self.args["use_recurrent_policy"] = True
            self.args["use_naive_recurrent_policy"] = False
        elif self.args["policy_algo"] == "mappo":
            self.args["use_recurrent_policy"] = False 
            self.args["use_naive_recurrent_policy"] = False
        elif self.args["policy_algo"] == "ippo":
            self.use_centralized_V = False
        else:
            raise NotImplementedError("Bad param given for policy_algo.")

        # Policy
        if not self.use_centralized_V:
            shared_obs_dim = obs_dim
        self.policy = R_MAPPOPolicy(
            self.args, obs_dim, shared_obs_dim, act_space, device)

        # Train algorithm
        self.trainer = R_MAPPOTrainAlgo(self.args, self.policy, device)

        # Replay buffer
        self.buffer = SharedReplayBuffer(
            self.args, n_agents, obs_dim, shared_obs_dim, act_space)

    def prep_rollout(self, device=None):
        if device is None:
            device = self.train_device
        self.trainer.prep_rollout(device)

    def prep_training(self):
        self.trainer.prep_training(self.train_device)

    def reset_buffer(self):
        self.buffer.reset_episode()

    def store_obs(self, obs, shared_obs):
        """
        Store observations in replay buffer.
        :param obs: (np.ndarray) Observations for each agent, 
            dim=(n_parallel_envs, n_agents, obs_dim).
        :param shared_obs: (np.ndarray) Centralised observations, 
            dim=(n_parallel_envs, n_agents, shared_obs_dim).
        """
        if not self.use_centralized_V:
            shared_obs = obs.copy()
        # else:
            # Repeat for all agents
            # shared_obs = np.repeat(shared_obs, self.n_agents, axis=0).reshape(
            #     self.n_parallel_envs, self.n_agents, -1)
        self.buffer.insert_obs(obs, shared_obs)

    @torch.no_grad()
    def get_actions(self):
        shared_obs, obs, rnn_states, critic_rnn_states, masks \
            = self.buffer.get_act_params()

        values, actions, action_log_probs, rnn_states, critic_rnn_states \
            = self.trainer.policy.get_actions(
                np.concatenate(shared_obs),
                np.concatenate(obs), 
                np.concatenate(rnn_states), 
                np.concatenate(critic_rnn_states), 
                np.concatenate(masks))

        values = np.reshape(
            torch2numpy(values), (self.n_parallel_envs, self.n_agents, -1))
        actions = np.reshape(
            torch2numpy(actions), (self.n_parallel_envs, self.n_agents, -1))
        action_log_probs = np.reshape(
            torch2numpy(action_log_probs), 
            (self.n_parallel_envs, self.n_agents, -1))
        rnn_states = np.reshape(
            torch2numpy(rnn_states), 
            (self.n_parallel_envs, self.n_agents, self.recurrent_N, -1))
        critic_rnn_states = np.reshape(
            torch2numpy(critic_rnn_states), 
            (self.n_parallel_envs, self.n_agents, self.recurrent_N, -1))

        return values, actions, action_log_probs, rnn_states, \
               critic_rnn_states

    def store_act(self, rewards, dones, values, actions, action_log_probs, 
                  rnn_states, rnn_states_critic):
        rnn_states[dones == True] = np.zeros(
            ((dones == True).sum(), 
             self.recurrent_N, 
             self.args["hidden_size"]),
            dtype=np.float32)
        rnn_states_critic[dones == True] = np.zeros(
            ((dones == True).sum(), 
             self.recurrent_N, 
             self.args["hidden_size"]),
            dtype=np.float32)
        masks = np.ones(
            (self.n_parallel_envs, self.n_agents, 1), dtype=np.float32)
        masks[dones == True] = np.zeros(
            ((dones == True).sum(), 1), dtype=np.float32)

        self.buffer.insert_act(
            rnn_states,
            rnn_states_critic,
            actions,
            action_log_probs,
            values,
            rewards,
            masks)

    @torch.no_grad()
    def compute_last_value(self):
        next_value = self.trainer.policy.get_values(
            np.concatenate(self.buffer.shared_obs[-1]),
            np.concatenate(self.buffer.rnn_states[-1]),
            np.concatenate(self.buffer.masks[-1]))
        
        next_value = np.reshape(
            torch2numpy(next_value), (self.n_parallel_envs, self.n_agents, -1))

        self.buffer.compute_returns(
            next_value, self.trainer.value_normalizer)

    def train(self, warmup=False):
        # Compute last value
        self.compute_last_value()
        # Train
        self.prep_training()
        train_infos = self.trainer.train(
            self.buffer, warmup=warmup)
        return train_infos

    def get_save_dict(self):
        self.prep_rollout("cpu")
        params = {
            "actor": self.trainer.policy.actor.state_dict(),
            "critic": self.trainer.policy.critic.state_dict()
        }
        if self.trainer._use_valuenorm:
            params["vnorm"] = self.trainer.value_normalizer.state_dict()
        save_dict = {
            "agents_params": params
        }
        return save_dict

    def load_params(self, params):
        self.trainer.policy.actor.load_state_dict(
            params["actor"])
        self.trainer.policy.critic.load_state_dict(
            params["critic"])
        if self.trainer._use_valuenorm:
            self.trainer.value_normalizer.load_state_dict(
                params["vnorm"])

    def save(self, path):
        save_dict = self.get_save_dict()
        torch.save(save_dict, path)

