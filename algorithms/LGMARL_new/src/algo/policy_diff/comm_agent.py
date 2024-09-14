import torch
from torch import nn

from .utils import update_linear_schedule, update_lr
from src.algo.nn_modules.mlp import MLPNetwork
from src.algo.nn_modules.rnn import RNNLayer
from src.algo.nn_modules.distributions import DiagGaussian, Categorical
from src.algo.nn_modules.utils import init_
from .comm_pol import CommPolicy


class CommAgent(nn.Module):

    def __init__(
            self, args, parser, n_agents, obs_dim, joint_obs_dim, act_dim, 
            device):
        super(CommAgent, self).__init__()
        self.lr = args.lr
        self.comm_type = args.comm_type
        self.warming_up = False
        self.context_dim = args.context_dim
        self.device = device

        # Common encoders
        self.obs_in = MLPNetwork(
            obs_dim, 
            args.hidden_dim, 
            args.hidden_dim)
        self.obs_encoder = RNNLayer(
            args.hidden_dim, 
            args.hidden_dim, 
            args.policy_recurrent_N)

        self.joint_obs_in = MLPNetwork(
            joint_obs_dim, 
            args.hidden_dim, 
            args.hidden_dim)
        self.joint_obs_encoder = RNNLayer(
            args.hidden_dim, 
            args.hidden_dim, 
            args.policy_recurrent_N)

        self.comm_pol = CommPolicy(args, n_agents, obs_dim)
        
        if "no_comm" in self.comm_type:
            act_pol_input = args.hidden_dim
            act_val_input = args.hidden_dim
        else:
            act_pol_input = args.hidden_dim * 2
            act_val_input = args.hidden_dim * 2

        # Act MAPPO 
        self.act_pol = nn.Sequential(
            MLPNetwork(
                act_pol_input, 
                args.hidden_dim, 
                args.hidden_dim, 
                n_hidden_layers=1,
                out_activation_fn="relu"),
            Categorical(args.hidden_dim, act_dim))
        self.act_val = nn.Sequential(
            MLPNetwork(
                act_val_input, 
                args.hidden_dim, 
                args.hidden_dim, 
                n_hidden_layers=1,
                out_activation_fn="relu"),
            init_(nn.Linear(args.hidden_dim, 1)))

        self.actor_optim = torch.optim.Adam(
            list(self.obs_in.parameters()) +
            list(self.obs_encoder.parameters()) +
            list(self.act_pol.parameters()),
            lr=self.lr,
            eps=args.opti_eps,
            weight_decay=args.weight_decay)

        self.comm_optim = torch.optim.Adam(
            self.comm_pol.parameters(),
            lr=self.lr,
            eps=args.opti_eps,
            weight_decay=args.weight_decay)

        self.critic_optim = torch.optim.Adam(
            list(self.joint_obs_in.parameters()) +
            list(self.joint_obs_encoder.parameters()) +
            list(self.act_val.parameters()),
            lr=self.lr,
            eps=args.opti_eps,
            weight_decay=args.weight_decay)

    def set_device(self, device):
        self.device = device

    def forward_comm(
            self, obs, joint_obs, obs_rnn_states, joint_obs_rnn_states, 
            masks, perfect_messages, deterministic=False, 
            eval_comm_actions=None):
        """
        Forward pass on the communication actor-critic.
        :param obs (torch.Tensor): observations.
        :param joint_obs: (torch.Tensor) joint observations.
        :param obs_rnn_states: (torch.Tensor) hidden states of the 
            obs encoder.
        :param joint_obs_rnn_states: (torch.Tensor) hidden states of the 
            joint obs encoder.
        :param masks: (torch.Tensor) mask tensor denoting if hidden states 
            should be reinitialized to zeros.
        :param perfect_messages: (torch.Tensor) perfect messages to use when 
            learning to use language
        :param deterministic: (boolean) whether generate deterministic outputs.
        :param eval_comm_actions: (torch.Tensor) comm actions to evaluate, if 
            given.

        :return messages: (list) generated messages.
        :return enc_obs: (torch.Tensor) encoded observation.
        :return enc_joint_obs: (torch.Tensor) encoded joint observation.
        :return comm_actions: (torch.Tensor) communication actions.
        :return comm_action_log_probs: (torch.Tensor) log-probabilities of 
            communication actions.
        :return comm_values: (torch.Tensor) communication value predictions.
        :return new_obs_rnn_states: (torch.Tensor) new hidden states of 
            the obs encoder.
        :return new_joint_obs_rnn_states: (torch.Tensor) new hidden states of 
            the joint obs encoder.
        """
        # Encode obs and joint obs
        enc_obs = self.obs_in(obs)
        enc_obs, new_obs_rnn_states = self.obs_encoder(
            enc_obs, obs_rnn_states, masks)
        enc_joint_obs = self.joint_obs_in(joint_obs)
        enc_joint_obs, new_joint_obs_rnn_states = self.joint_obs_encoder(
            enc_joint_obs, joint_obs_rnn_states, masks)

        messages, comm_actions, comm_action_log_probs, comm_values, \
            eval_comm_action_log_probs, eval_comm_dist_entropy \
            = self.comm_pol.gen_comm(enc_obs, perfect_messages, obs)

        return messages, enc_obs, enc_joint_obs, comm_actions, \
            comm_action_log_probs, comm_values, new_obs_rnn_states, \
            new_joint_obs_rnn_states, eval_comm_action_log_probs, \
            eval_comm_dist_entropy
        
    def forward_act(
            self, messages, enc_obs, enc_joint_obs, comm_rnn_states, masks, 
            deterministic=False, eval_actions=None):
        """
        Forward pass on the action actor-critic.
        :param messages (torch.Tensor): incoming messages (already encoded if 
            using language).
        :param enc_obs: (torch.Tensor) encoded observations.
        :param enc_joint_obs: (torch.Tensor) encoded joint observations.
        :param comm_rnn_states: (torch.Tensor) hidden states of the 
            communication encoder.
        :param masks: (torch.Tensor) mask tensor denoting if hidden states 
            should be reinitialized to zeros.
        :param deterministic: (boolean) whether generate deterministic outputs.

        :return actions: (torch.Tensor) actions.
        :return action_log_probs: (torch.Tensor) log-probabilities of actions.
        :return values: (torch.Tensor) value predictions.
        :return new_comm_rnn_states: (torch.Tensor) new hidden states of 
            the communication encoder.
        """
        # Encode messages
        pol_input, val_input, new_comm_rnn_states = self.comm_pol.enc_comm(
            enc_obs, enc_joint_obs, messages, comm_rnn_states, masks)

        # Compute actions
        action_logits = self.act_pol(pol_input)
        if deterministic:
            actions = action_logits.mode()
        else:
            actions = action_logits.sample()
        action_log_probs = action_logits.log_probs(actions)

        # Get values
        values = self.act_val(val_input)

        if eval_actions is not None:
            eval_action_log_probs = action_logits.log_probs(eval_actions)
            eval_dist_entropy = action_logits.entropy().mean()
        else:
            eval_action_log_probs = None
            eval_dist_entropy = None
        
        return actions, action_log_probs, values, new_comm_rnn_states, \
            eval_action_log_probs, eval_dist_entropy

    def warmup_lr(self, warmup):
        if warmup != self.warming_up:
            lr = self.lr * 0.01 if warmup else self.lr
            update_lr(self.actor_optim, lr)
            update_lr(self.critic_optim, lr)
            self.warming_up = warmup