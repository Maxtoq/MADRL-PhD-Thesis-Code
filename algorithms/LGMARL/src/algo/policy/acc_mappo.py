import torch
import numpy as np

from .policy import ACCPolicy
from .buffer import ACC_ReplayBuffer
from .utils import update_linear_schedule, update_lr, get_shape_from_obs_space, torch2numpy


class ACC_MAPPO:

    def __init__(self, args, lang_learner, n_agents, obs_space, 
                 shared_obs_space, act_space, device):
        self.args = args
        self.lang_learner = lang_learner
        self.n_agents = n_agents
        self.device = device
        self.context_dim = args.context_dim
        self.n_envs = args.n_parallel_envs
        self.recurrent_N = args.policy_recurrent_N
        self.hidden_dim = args.hidden_dim
        self.lr = args.lr
        self.warming_up = False

        obs_dim = get_shape_from_obs_space(obs_space[0]) + self.context_dim
        shared_obs_dim = get_shape_from_obs_space(shared_obs_space[0]) \
                            + self.context_dim
        act_dim = act_space.n
        self.policy = ACCPolicy(args, obs_dim, shared_obs_dim, act_dim)

        self.rl_optim = torch.optim.Adam(
            self.policy.parameters(), 
            lr=self.lr, 
            eps=args.opti_eps, 
            weight_decay=args.weight_decay)

        self.buffer = ACC_ReplayBuffer(
            self.args, 
            n_agents, 
            obs_dim, 
            shared_obs_dim, 
            act_dim, 
            self.context_dim)

    def lr_decay(self, episode, episodes):
        """
        Decay the actor and critic learning rates.
        :param episode: (int) current training episode.
        :param episodes: (int) total number of training episodes.
        """
        update_linear_schedule(
            self.rl_optim, episode, episodes, self.lr)

    def warmup_lr(self, warmup):
        if warmup != self.warming_up:
            lr = self.lr * 0.01 if warmup else self.lr
            update_lr(self.rl_optim, lr)
            self.warming_up = warmup

    def prep_rollout(self, device=None):
        if device is not None:
            self.device = device
        self.policy.act_comm.eval()
        self.policy.act_comm.to(self.device)
        self.policy.critic.eval()
        self.policy.critic.to(self.device)

    def prep_training(self, device=None):
        if device is not None:
            self.device = device
        self.policy.act_comm.train()
        self.policy.act_comm.to(self.device)
        self.policy.critic.train()
        self.policy.critic.to(self.device)

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

