import torch
import numpy as np

from .r_actor_critic import R_Actor, R_Critic
from .utils import update_linear_schedule

##########################################################################
# Code modified from https://github.com/marlbenchmark/on-policy
##########################################################################


class R_MAPPOPolicy:
    """
    MAPPO Policy  class. Wraps actor and critic networks to compute actions and value function predictions.

    :param args: (dict) arguments containing relevant model and policy information.
    TODO: changer les types des obs et act space
    :param obs_dim: (int) observation dimension.
    :param cent_obs_dim: (int) value function input dimension (centralized input for MAPPO, decentralized for IPPO).
    :param action_dim: (int) action dimension.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """

    def __init__(self, 
            args, obs_dim, cent_obs_dim, act_space, 
            device=torch.device("cpu")):
        self.lr = args["lr"]
        self.critic_lr = args["critic_lr"]
        self.opti_eps = args["opti_eps"]
        self.weight_decay = args["weight_decay"]
        self.warming_up = False

        self.actor = R_Actor(args, obs_dim, act_space, device)
        self.critic = R_Critic(args, cent_obs_dim, device)

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=self.lr, eps=self.opti_eps,
            weight_decay=self.weight_decay)
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=self.critic_lr,
            eps=self.opti_eps,
            weight_decay=self.weight_decay)

    def lr_decay(self, episode, episodes):
        """
        Decay the actor and critic learning rates.
        :param episode: (int) current training episode.
        :param episodes: (int) total number of training episodes.
        """
        update_linear_schedule(
            self.actor_optimizer, episode, episodes, self.lr)
        update_linear_schedule(
            self.critic_optimizer, episode, episodes, self.critic_lr)

    def warmup_lr(self, warmup):
        if warmup != self.warming_up:
            lr = self.lr * 0.01 if warmup else self.lr
            for param_group in self.actor_optimizer.param_groups:
                param_group['lr'] = lr
            for param_group in self.critic_optimizer.param_groups:
                param_group['lr'] = lr
            self.warming_up = warmup

    def get_actions(self, 
            cent_obs, obs, rnn_states_actor, rnn_states_critic, masks,
            available_actions=None, deterministic=False):
        """
        Compute actions and value function predictions for the given inputs.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
        :param rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :param available_actions: (np.ndarray) denotes which actions are
            available to agent (if None, all actions available).
        :param deterministic: (bool) whether the action should be mode of 
            distribution or should be sampled.

        :return values: (torch.Tensor) value function predictions.
        :return actions: (torch.Tensor) actions to take.
        :return action_log_probs: (torch.Tensor) log probabilities of chosen actions.
        :return rnn_states_actor: (torch.Tensor) updated actor network RNN states.
        :return rnn_states_critic: (torch.Tensor) updated critic network RNN states.
        """
        actions, action_log_probs, rnn_states_actor = self.actor(
            obs, rnn_states_actor, masks, available_actions, 
            deterministic)

        values, rnn_states_critic = self.critic(
            cent_obs, rnn_states_critic, masks)
        return values, actions, action_log_probs, rnn_states_actor, \
            rnn_states_critic

    def get_values(self, cent_obs, rnn_states_critic, masks):
        """
        Get value function predictions.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for
            critic.
        :param masks: (np.ndarray) denotes points at which RNN states should be 
            reset.

        :return values: (torch.Tensor) value function predictions.
        """
        values, _ = self.critic(cent_obs, rnn_states_critic, masks)
        return values

    def evaluate_actions(self, 
            cent_obs, obs, rnn_states_actor, rnn_states_critic, 
            action, masks, available_actions=None, active_masks=None):
        """
        Get action logprobs / entropy and value function predictions for actor
        update.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for 
            actor.
        :param rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for
            critic.
        :param action: (np.ndarray) actions whose log probabilites and entropy
            to compute.
        :param masks: (np.ndarray) denotes points at which RNN states should be
            reset.
        :param available_actions: (np.ndarray) denotes which actions are 
            available to agent (if None, all actions available).
        :param active_masks: (torch.Tensor) denotes whether an agent is active 
            or dead.

        :return values: (torch.Tensor) value function predictions.
        :return action_log_probs: (torch.Tensor) log probabilities of the input
            actions.
        :return dist_entropy: (torch.Tensor) action distribution entropy for
            the given inputs.
        """
        action_log_probs, dist_entropy = self.actor.evaluate_actions(
            obs, rnn_states_actor, action, masks, available_actions, 
            active_masks)

        values, _ = self.critic(cent_obs, rnn_states_critic, masks)
        return values, action_log_probs, dist_entropy

    def act(self, 
            obs, rnn_states_actor, masks, 
            available_actions=None, deterministic=False):
        """
        Compute actions using the given inputs.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for
            actor.
        :param masks: (np.ndarray) denotes points at which RNN states should be
            reset.
        :param available_actions: (np.ndarray) denotes which actions are 
            available to agent (if None, all actions available)
        :param deterministic: (bool) whether the action should be mode of 
            distribution or should be sampled.
        """
        actions, _, rnn_states_actor = self.actor(
            obs, rnn_states_actor, masks, available_actions, deterministic)
        return actions, rnn_states_actor