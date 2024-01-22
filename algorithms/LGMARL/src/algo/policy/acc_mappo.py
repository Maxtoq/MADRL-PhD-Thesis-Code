import torch
import numpy as np

from .policy import ACCPolicy
from .acc_mappo_trainer import ACC_MAPPOTrainAlgo
from .buffer import ACC_ReplayBuffer
from .utils import get_shape_from_obs_space, torch2numpy


class ACC_MAPPO:

    def __init__(self, args, lang_learner, n_agents, obs_space, 
                 shared_obs_space, act_space, device):
        self.args = args
        self.lang_learner = lang_learner
        self.n_agents = n_agents
        self.train_device = device
        self.context_dim = args.context_dim
        self.n_envs = args.n_parallel_envs
        self.recurrent_N = args.policy_recurrent_N
        self.hidden_dim = args.hidden_dim
        self.lr = args.lr

        self.device = self.train_device

        obs_dim = get_shape_from_obs_space(obs_space[0]) + self.context_dim
        shared_obs_dim = get_shape_from_obs_space(shared_obs_space[0]) \
                            + self.context_dim
        act_dim = act_space.n
        self.policy = ACCPolicy(args, obs_dim, shared_obs_dim, act_dim)

        self.trainer = ACC_MAPPOTrainAlgo(args, self.policy, device)

        self.buffer = ACC_ReplayBuffer(
            self.args, 
            n_agents, 
            obs_dim, 
            shared_obs_dim, 
            1, 
            self.context_dim)

    def prep_rollout(self, device=None):
        self.device = self.train_device if device is None else device
        self.policy.act_comm.eval()
        self.policy.act_comm.to(self.device)
        self.policy.critic.eval()
        self.policy.critic.to(self.device)
        self.trainer.device = self.device
        if self.trainer.value_normalizer is not None:
            self.trainer.value_normalizer.to(self.device)

    def prep_training(self, device=None):
        if device is not None:
            self.train_device = device
        self.device = self.train_device
        self.policy.act_comm.train()
        self.policy.act_comm.to(self.train_device)
        self.policy.critic.train()
        self.policy.critic.to(self.train_device)
        self.trainer.device = self.train_device
        if self.trainer.value_normalizer is not None:
            self.trainer.value_normalizer.to(self.train_device)

    def reset_buffer(self):
        self.buffer.reset_episode()

    def store_obs(self, obs, shared_obs):
        """
        Store observations in replay buffer.
        :param obs: (np.ndarray) Observations for each agent, 
            dim=(n_envs, n_agents, obs_dim).
        :param shared_obs: (np.ndarray) Centralised observations, 
            dim=(n_envs, n_agents, shared_obs_dim).
        """
        self.buffer.insert_obs(obs, shared_obs)

    def store_act(self, rewards, dones, values, actions, action_log_probs, 
                  comm_actions, comm_action_log_probs, rnn_states, 
                  rnn_states_critic):
        rnn_states[dones == True] = np.zeros(
            ((dones == True).sum(), 
             self.recurrent_N, 
             self.hidden_dim),
            dtype=np.float32)
        rnn_states_critic[dones == True] = np.zeros(
            ((dones == True).sum(), 
             self.recurrent_N, 
             self.hidden_dim),
            dtype=np.float32)
        masks = np.ones((self.n_envs, self.n_agents, 1), dtype=np.float32)
        masks[dones == True] = np.zeros(
            ((dones == True).sum(), 1), dtype=np.float32)

        self.buffer.insert_act(
            rnn_states,
            rnn_states_critic,
            actions,
            action_log_probs,
            comm_actions,
            comm_action_log_probs,
            values,
            rewards,
            masks)

    @torch.no_grad()
    def get_actions(self):
        obs, shared_obs, rnn_states, critic_rnn_states, masks \
            = self.buffer.get_act_params()

        obs = torch.from_numpy(np.concatenate(obs)).to(self.device)
        shared_obs = torch.from_numpy(np.concatenate(shared_obs)).to(self.device)
        rnn_states = torch.from_numpy(np.concatenate(rnn_states)).to(self.device)
        critic_rnn_states = torch.from_numpy(
            np.concatenate(critic_rnn_states)).to(self.device)
        masks = torch.from_numpy(np.concatenate(masks)).to(self.device)

        values, actions, action_log_probs, comm_actions, comm_action_log_probs, \
            rnn_states, critic_rnn_states = self.policy(
                obs, shared_obs, rnn_states, critic_rnn_states, masks)

        values = np.reshape(
            torch2numpy(values), (self.n_envs, self.n_agents, -1))
        actions = np.reshape(
            torch2numpy(actions), (self.n_envs, self.n_agents, -1))
        action_log_probs = np.reshape(
            torch2numpy(action_log_probs), 
            (self.n_envs, self.n_agents, -1))
        comm_actions = np.reshape(
            torch2numpy(comm_actions), (self.n_envs, self.n_agents, -1))
        comm_action_log_probs = np.reshape(
            torch2numpy(comm_action_log_probs), 
            (self.n_envs, self.n_agents, -1))
        rnn_states = np.reshape(
            torch2numpy(rnn_states), 
            (self.n_envs, self.n_agents, self.recurrent_N, -1))
        critic_rnn_states = np.reshape(
            torch2numpy(critic_rnn_states), 
            (self.n_envs, self.n_agents, self.recurrent_N, -1))

        return values, actions, action_log_probs, comm_actions, \
                comm_action_log_probs, rnn_states, critic_rnn_states

    @torch.no_grad()
    def _compute_last_value(self):
        next_value = self.policy.get_values(
            torch.from_numpy(np.concatenate(
                self.buffer.shared_obs[-1])).to(self.device),
            torch.from_numpy(np.concatenate(
                self.buffer.rnn_states[-1])).to(self.device),
            torch.from_numpy(np.concatenate(
                self.buffer.masks[-1])).to(self.device))
        
        next_value = np.reshape(
            torch2numpy(next_value), (self.n_envs, self.n_agents, -1))

        self.buffer.compute_returns(
            next_value, self.trainer.value_normalizer)

    def train(self, warmup=False, train_comm_head=True):
        # Compute last value
        self._compute_last_value()
        # Train
        self.prep_training()
        self.policy.warmup_lr(warmup)
        losses = self.trainer.train(self.buffer, train_comm_head)
        return losses

