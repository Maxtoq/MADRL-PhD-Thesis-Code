import torch
from torch import nn

from .actor_communicator import ActorCommunicator
from .critic import Critic
from .utils import update_linear_schedule, update_lr


class ACC_Agent(nn.Module):

    def __init__(self, args, obs_dim, shared_obs_dim, act_dim):
        super(ACC_Agent, self).__init__()
        self.lr = args.lr
        self.warming_up = False

        # Actor-Communicator
        self.act_comm = ActorCommunicator(args, obs_dim, act_dim)

        # Critic
        self.critic = Critic(args, shared_obs_dim)

        self.act_comm_optim = torch.optim.Adam(
            self.act_comm.parameters(), 
            lr=self.lr, 
            eps=args.opti_eps, 
            weight_decay=args.weight_decay)
        self.critic_optim = torch.optim.Adam(
            self.critic.parameters(), 
            lr=self.lr, 
            eps=args.opti_eps, 
            weight_decay=args.weight_decay)

    def lr_decay(self, episode, episodes):
        """
        Decay the actor and critic learning rates.
        :param episode: (int) current training episode.
        :param episodes: (int) total number of training episodes.
        """
        update_linear_schedule(
            self.act_comm_optim, episode, episodes, self.lr)
        update_linear_schedule(
            self.critic_optim, episode, episodes, self.lr)

    def warmup_lr(self, warmup):
        if warmup != self.warming_up:
            lr = self.lr * 0.01 if warmup else self.lr
            update_lr(self.act_comm_optim, lr)
            update_lr(self.critic_optim, lr)
            self.warming_up = warmup

    def forward(self, obs, shared_obs, act_rnn_states, critic_rnn_states, masks):
        actions, action_log_probs, comm_actions, comm_action_log_probs, \
            act_rnn_states = self.act_comm(obs, act_rnn_states, masks)

        values, critic_rnn_states = self.critic(
            shared_obs, critic_rnn_states, masks)

        return values, actions, action_log_probs, comm_actions, \
            comm_action_log_probs, act_rnn_states, critic_rnn_states

    def get_values(self, shared_obs, rnn_states_critic, masks):
        """
        Get value function predictions.
        :param shared_obs (torch.Tensor): centralized input to the critic.
        :param rnn_states_critic: (torch.Tensor) if critic is RNN, RNN states for
            critic.
        :param masks: (torch.Tensor) denotes points at which RNN states should be 
            reset.

        :return values: (torch.Tensor) value function predictions.
        """
        values, _ = self.critic(shared_obs, rnn_states_critic, masks)
        return values

    def evaluate_actions(self, 
            obs, shared_obs, rnn_states_actor, rnn_states_critic, 
            env_actions, comm_actions, masks, eval_comm=True):
        """
        Get action logprobs / entropy and value function predictions for actor
        update.
        :param obs: (torch.Tensor) local agent inputs to the actor.
        :param shared_obs: (torch.Tensor) centralized input to the critic.
        :param rnn_states_actor: (torch.Tensor) if actor is RNN, RNN states 
            for actor.
        :param rnn_states_critic: (torch.Tensor) if critic is RNN, RNN states
            for critic.
        :param env_actions: (torch.Tensor) environment actions whose log 
            probabilites and entropy to compute.
        :param comm_actions: (torch.Tensor) communication actions whose log 
            probabilites and entropy to compute.
        :param masks: (torch.Tensor) denotes points at which RNN states should
            be reset.
        :param eval_comm: (bool) whether to compute comm_actions probs.

        :return values: (torch.Tensor) value function predictions.
        :return env_action_log_probs: (torch.Tensor) log probabilities of the
            environment actions.
        :return env_dist_entropy: (torch.Tensor) environment action 
            distribution entropy for the given inputs.
        :return comm_action_log_probs: (torch.Tensor) log probabilities of the
            communication actions.
        :return comm_dist_entropy: (torch.Tensor) communication action 
            distribution entropy for the given inputs.
        """
        env_action_log_probs, env_dist_entropy, comm_action_log_probs, \
            comm_dist_entropy = self.act_comm.evaluate_actions(
                obs, rnn_states_actor, env_actions, comm_actions, masks, 
                eval_comm)

        values, _ = self.critic(shared_obs, rnn_states_critic, masks)

        return values, env_action_log_probs, env_dist_entropy, \
                comm_action_log_probs, comm_dist_entropy