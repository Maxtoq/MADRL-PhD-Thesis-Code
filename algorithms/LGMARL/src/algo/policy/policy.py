import torch
from torch import nn

from .actor_communicator import ActorCommunicator
from .critic import Critic


class ACCPolicy(nn.Module):

    def __init__(self, args, obs_dim, shared_obs_dim, act_dim):
        super(ACCPolicy, self).__init__()
        # Actor-Communicator
        self.act_comm = ActorCommunicator(args, obs_dim, act_dim)

        # Critic
        self.critic = Critic(args, shared_obs_dim)

    def forward(self, obs, shared_obs, act_rnn_states, critic_rnn_states, masks):
        actions, action_log_probs, comm_actions, comm_action_log_probs, \
            act_rnn_states = self.act_comm(obs, act_rnn_states, masks)

        values, critic_rnn_states = self.critic(
            shared_obs, critic_rnn_states, masks)

        return values, actions, action_log_probs, comm_actions, \
            comm_action_log_probs, act_rnn_states, critic_rnn_states
