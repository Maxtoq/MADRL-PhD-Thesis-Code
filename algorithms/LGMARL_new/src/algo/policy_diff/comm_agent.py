import torch
from torch import nn

from .utils import update_linear_schedule, update_lr
from src.algo.nn_modules.mlp import MLPNetwork
from src.algo.nn_modules.rnn import RNNLayer
from src.algo.nn_modules.distributions import DiagGaussian, Categorical
from src.algo.nn_modules.utils import init


def init_(m):
    return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0))


class Comm_Agent(nn.Module):

    def __init__(
            self, args, n_agents, obs_dim, joint_obs_dim, act_dim, device):
        super(Comm_Agent, self).__init__()
        self.lr = args.lr
        self.comm_type = args.comm_type
        self.warming_up = False
        self.context_dim = args.context_dim
        self.device = device

        # Common encoders
        self.obs_in = MLPNetwork(
            obs_dim, args.hidden_dim, args.hidden_dim)
        self.obs_encoder = RNNLayer(
            args.hidden_dim, args.hidden_dim, args.policy_recurrent_N)

        self.joint_obs_in = MLPNetwork(
            joint_obs_dim, args.hidden_dim, args.hidden_dim)
        self.joint_obs_encoder = RNNLayer(
            args.hidden_dim, args.hidden_dim, args.policy_recurrent_N)
        
        # Comm 
        # self.comm_pol = nn.Sequential(
        #     MLPNetwork(
        #         args.hidden_dim, 
        #         args.hidden_dim, 
        #         args.hidden_dim, 
        #         out_activation_fn="relu"),
        #     DiagGaussian(args.hidden_dim, args.context_dim))
        self.comm_pol = MLPNetwork(
                args.hidden_dim, 
                args.context_dim, 
                args.hidden_dim, 
                out_activation_fn="tanh")

        self.comm_val = nn.Sequential(
            MLPNetwork(
                args.hidden_dim, 
                args.hidden_dim, 
                args.hidden_dim, 
                out_activation_fn="relu"),
            init_(nn.Linear(args.hidden_dim, 1)))

        if self.comm_type == "emergent_continuous":
            in_comm_enc = n_agents * args.context_dim
        else:
            in_comm_enc = args.context_dim
        self.comm_encoder = RNNLayer(
            in_comm_enc, args.hidden_dim, args.policy_recurrent_N)

            # if self.comm_type == "emergent_continuous":
        # self.message_encoder = init_(nn.Linear(
        #     n_agents * args.context_dim, args.context_dim))

        
        if self.comm_type in ["emergent_continuous", "language", "perfect"]:
            act_pol_input = args.hidden_dim * 2
            act_val_input = args.hidden_dim * 2
        elif self.comm_type == "no_comm":
            act_pol_input = args.hidden_dim
            act_val_input = args.hidden_dim
            # self.comm_encoder = None
            # self.comm_pol = None
            # self.comm_val = None

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
            list(self.comm_pol.parameters()) +
            # list(self.message_encoder.parameters()) +
            list(self.comm_encoder.parameters()) + 
            list(self.act_pol.parameters()),
            lr=self.lr,
            eps=args.opti_eps,
            weight_decay=args.weight_decay)

        self.critic_optim = torch.optim.Adam(
            list(self.joint_obs_in.parameters()) +
            list(self.joint_obs_encoder.parameters()) +
            list(self.comm_val.parameters()) +
            list(self.comm_encoder.parameters()) + 
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

        # Get comm actions and values
        if self.comm_type == "no_comm":
            comm_actions = torch.zeros(obs.shape[0], self.context_dim)
            comm_action_log_probs = torch.zeros(obs.shape[0], 1)
            comm_values = torch.zeros(obs.shape[0], 1)
            messages = None
            eval_comm_action_log_probs = None
            eval_comm_dist_entropy = None
            lang_obs_enc = None

        elif self.comm_type == "emergent_continuous":
            # Get comm_action
            # comm_action_logits = self.comm_pol(enc_obs)
            # if deterministic:
            #     comm_actions = comm_action_logits.mode()
            # else:
            #     comm_actions = comm_action_logits.sample() 
            # comm_action_log_probs = comm_action_logits.log_probs(comm_actions)

            # if eval_comm_actions is not None:
            #     eval_comm_action_log_probs = comm_action_logits.log_probs(
            #         eval_comm_actions)
            #     eval_comm_dist_entropy = comm_action_logits.entropy().mean()
            # else:

            comm_actions = self.comm_pol(enc_obs)

            # Get comm_value
            # comm_values = self.comm_val(enc_joint_obs)
            eval_comm_action_log_probs = None
            eval_comm_dist_entropy = None
            comm_action_log_probs = torch.zeros(obs.shape[0], 1)
            comm_values = torch.zeros(obs.shape[0], 1)

            messages = comm_actions.clone()

            lang_obs_enc = None

        elif self.comm_type == "perfect":
            # comm_actions = self.comm_pol(enc_obs).mode()
            comm_actions = self.comm_pol(enc_obs)
            comm_action_log_probs = torch.zeros(obs.shape[0], 1)
            comm_values = torch.zeros(obs.shape[0], 1)
            messages = perfect_messages
            eval_comm_action_log_probs = None
            eval_comm_dist_entropy = None
        else:
            raise NotImplementedError("Bad comm_type:" + self.comm_type)

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
        if self.comm_type == "no_comm":
            pol_input = enc_obs
            val_input = enc_joint_obs
            new_comm_rnn_states = torch.zeros_like(comm_rnn_states)
        elif self.comm_type == "emergent_continuous":
            # enc_mess = self.message_encoder(messages)

            comm_enc, new_comm_rnn_states = self.comm_encoder(
                messages, comm_rnn_states, masks)

            pol_input = torch.concatenate((comm_enc, enc_obs), dim=-1)
            val_input = torch.concatenate((comm_enc, enc_joint_obs), dim=-1)
        elif self.comm_type  == "perfect":
            comm_enc, new_comm_rnn_states = self.comm_encoder(
                messages, comm_rnn_states, masks)

            pol_input = torch.concatenate((comm_enc, enc_obs), dim=-1)
            val_input = torch.concatenate((comm_enc, enc_joint_obs), dim=-1)

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

    # def get_values(self, joint_obs, joint_obs_rnn_states, masks):
    #     """
    #     Get value function predictions.

    #     :return act_values: (torch.Tensor) action value predictions.
    #     :return comm_values: (torch.Tensor) communication value predictions.
    #     """
    #     enc_joint_obs = self.joint_obs_in(joint_obs)
    #     enc_joint_obs, new_joint_obs_rnn_states = self.joint_obs_encoder(
    #         enc_joint_obs, joint_obs_rnn_states, masks)

    #     if self.comm_type in ["emergent_continuous", "perfect"]:
    #         comm_values = self.comm_val(enc_joint_obs)
    #         act_val_input = torch.concatenate(
    #             (enc_joint_obs, torch.zeros_like(enc_joint_obs).to(self.device)),
    #             dim=-1)

    #     elif self.comm_type == "no_comm":
    #         act_val_input = enc_joint_obs

    #     # elif self.comm_type == "perfect":
    #     #     comm_values = self.comm_val(enc_joint_obs)
    #     #     act_val_input = 0

    #     values = self.act_val(act_val_input)

    #     if self.comm_type == "no_comm":
    #         comm_values = torch.zeros_like(values)

    #     return values, comm_values
        
    # def evaluate_actions(
    #         self, obs, joint_obs, obs_enc_rnn_states, joint_obs_enc_rnn_states, 
    #         comm_enc_rnn_states, actions, comm_actions, masks):
    #     """
    #     Get action logprobs / entropy and value function predictions for actor
    #     update.
    #     :param obs: (torch.Tensor) local agent inputs to the actor.
    #     :param joint_obs: (torch.Tensor) centralized input to the critic.
    #     :param obs_enc_rnn_states: (torch.Tensor) 
    #     :param joint_obs_enc_rnn_states: (torch.Tensor) 
    #     :param comm_enc_rnn_states: (torch.Tensor) 
    #     :param actions: (torch.Tensor) environment actions whose log 
    #         probabilites and entropy to compute.
    #     :param comm_actions: (torch.Tensor) communication actions whose log 
    #         probabilites and entropy to compute.
    #     :param masks: (torch.Tensor) denotes points at which RNN states should
    #         be reset.

    #     :return act_values: (torch.Tensor) action value predictions.
    #     :return comm_values: (torch.Tensor) communication value predictions.
    #     :return env_action_log_probs: (torch.Tensor) log probabilities of the
    #         environment actions.
    #     :return env_dist_entropy: (torch.Tensor) environment action 
    #         distribution entropy for the given inputs.
    #     :return comm_action_log_probs: (torch.Tensor) log probabilities of the
    #         communication actions.
    #     :return comm_dist_entropy: (torch.Tensor) communication action 
    #         distribution entropy for the given inputs.

    #     """
    #     # env_action_log_probs, env_dist_entropy, comm_action_log_probs, \
    #     #     comm_dist_entropy, comm_actions = self.act_comm.evaluate_actions(
    #     #         obs, rnn_states_actor, env_actions, comm_actions, masks)

    #     # act_values, comm_values, _, obs_encs = self.critic(
    #     #     shared_obs, rnn_states_critic, masks, get_obs_encs=True)

    #     # Encode obs and joint obs
    #     enc_obs, new_obs_rnn_states = self.obs_encoder(
    #         obs, obs_enc_rnn_states, masks)
    #     enc_joint_obs, new_joint_obs_rnn_states = self.joint_obs_encoder(
    #         joint_obs, joint_obs_enc_rnn_states, masks)

    #     if self.comm_type == "no_comm":
    #         comm_actions = torch.zeros(obs.shape[0], self.context_dim)
    #         comm_action_log_probs = torch.zeros(obs.shape[0], 1)
    #         comm_dist_entropy = torch.zeros(obs.shape[0], 1)
    #         comm_values = torch.zeros(obs.shape[0], 1)

    #         pol_input = enc_obs
    #         val_input = enc_joint_obs
    #         new_comm_enc_rnn_states = torch.zeros_like(comm_enc_rnn_states)

    #     elif self.comm_type == "emergent_continuous":
    #         # redo communication step to have gradients flow between agent 
    #         comm_action_logits = self.comm_pol(enc_obs)
    #         new_comm_actions = comm_action_logits
    #         comm_action_log_probs = comm_action_logits.log_probs(comm_actions)

    #         # Get comm_value
    #         comm_values = self.comm_val(enc_joint_obs)

    #     # TODO handle comm
    #     # - emergent continuous (diff): 
    #     # - language: take messages sent during rollout to compute actions

    #     # Compute actions
    #     action_logits = self.act_pol(pol_input)
    #     action_log_probs = action_logits.log_probs(actions)
    #     dist_entropy = action_logits.entropy().mean()

    #     # Get values
    #     values = self.act_val(val_input)

    #     return values, comm_values, action_log_probs, dist_entropy, \
    #             comm_action_log_probs, comm_dist_entropy, comm_actions