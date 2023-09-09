import torch
import torch.nn as nn

from .utils import init, check
from .nn_modules.cnn import CNNBase
from .nn_modules.mlp import MLPBase, MLPNet
from .nn_modules.rnn import RNNLayer
from .nn_modules.act import ACTLayer
from .nn_modules.popart import PopArt

##########################################################################
# Code modified from https://github.com/marlbenchmark/on-policy
##########################################################################


class R_Actor(nn.Module):
    """
    Actor network class for MAPPO. Outputs actions given observations.
    :param args: (argparse.Namespace) arguments containing relevant model information.
    :param obs_dim: (int or tuple) observation dimension(s).
    :param act_dim: (int) action dim.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """
    def __init__(self, 
            args, obs_dim, act_space, device=torch.device("cpu")):
        super(R_Actor, self).__init__()
        self.hidden_size = args.hidden_size

        self._use_policy_active_masks = args.use_policy_active_masks
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._recurrent_N = args.recurrent_N
        self.tpdv = dict(dtype=torch.float32, device=device)

        #obs_dim = get_shape_from_obs_space(obs_space)
        self.obs_encoder = MLPBase(args, obs_dim)

        # # [LMC] Add dimension of context encoding
        # input_encoding_dim = self.hidden_size + context_dim

        # Add encoding layer
        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            self.rnn_encoder = RNNLayer(
                self.hidden_size, self.hidden_size, self._recurrent_N, 
                args.use_orthogonal)
        else:
            self.mlp_encoder = MLPNet(
                self.hidden_size, self.hidden_size, 1, 
                args.use_orthogonal, args.use_ReLU)

        self.act = ACTLayer(
            act_space, self.hidden_size, args.use_orthogonal, args.gain)

        self.to(device)

    def forward(self, 
            obs, rnn_states, masks, 
            available_actions=None, deterministic=False):
        """
        Compute actions from the given inputs.
        :param obs: (np.ndarray / torch.Tensor) observation inputs into network.
        :param rnn_states: (np.ndarray / torch.Tensor) if RNN network, hidden states for RNN.
        :param masks: (np.ndarray / torch.Tensor) mask tensor denoting if hidden states should be reinitialized to zeros.
        :param available_actions: (np.ndarray / torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
        :param deterministic: (bool) whether to sample from action distribution or return the mode.

        :return actions: (torch.Tensor) actions to take.
        :return action_log_probs: (torch.Tensor) log probabilities of taken actions.
        :return rnn_states: (torch.Tensor) updated RNN hidden states.
        """
        obs = torch.from_numpy(obs).to(**self.tpdv)
        rnn_states = torch.from_numpy(rnn_states).to(**self.tpdv)
        masks = torch.from_numpy(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = torch.from_numpy(available_actions).to(**self.tpdv)

        actor_features = self.obs_encoder(obs)

        # # [LMC] Concatenate incoming context encoding
        # actor_features = torch.cat((obs_enc, context), dim=-1)

        # Encode actor_features
        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            actor_features, rnn_states = self.rnn_encoder(
                actor_features, rnn_states, masks)
        else:
            actor_features = self.mlp_encoder(actor_features)

        actions, action_log_probs = self.act(
            actor_features, available_actions, deterministic)

        return actions, action_log_probs, rnn_states

    def evaluate_actions(self, 
            obs, rnn_states, action, masks, 
            available_actions=None, active_masks=None):
        """
        Compute log probability and entropy of given actions.
        :param obs: (torch.Tensor) observation inputs into network.
        :param action: (torch.Tensor) actions whose entropy and log probability to evaluate.
        :param rnn_states: (torch.Tensor) if RNN network, hidden states for RNN.
        :param masks: (torch.Tensor) mask tensor denoting if hidden states should be reinitialized to zeros.
        :param available_actions: (torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
        :param active_masks: (torch.Tensor) denotes whether an agent is active or dead.

        :return action_log_probs: (torch.Tensor) log probabilities of the input actions.
        :return dist_entropy: (torch.Tensor) action distribution entropy for the given inputs.
        """
        obs = torch.from_numpy(obs).to(**self.tpdv)
        context = torch.from_numpy(context).to(**self.tpdv)
        rnn_states = torch.from_numpy(rnn_states).to(**self.tpdv)
        action = torch.from_numpy(action).to(**self.tpdv)
        masks = torch.from_numpy(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = torch.from_numpy(available_actions).to(**self.tpdv)

        if active_masks is not None:
            active_masks = torch.from_numpy(active_masks).to(**self.tpdv)

        actor_features = self.obs_encoder(obs)

        #  # [LMC] Concatenate incoming context encoding
        # actor_features = torch.cat((obs_enc, context), dim=-1)

        # Encode actor_features
        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            actor_features, rnn_states = self.rnn_encoder(
                actor_features, rnn_states, masks)
        else:
            actor_features = self.mlp_encoder(actor_features)

        action_log_probs, dist_entropy = self.act.evaluate_actions(
            actor_features, action, available_actions,
            active_masks=active_masks if self._use_policy_active_masks 
                         else None)

        return action_log_probs, dist_entropy


class R_Critic(nn.Module):
    """
    Critic network class for MAPPO. Outputs value function predictions given 
    centralized input (MAPPO) or local observations (IPPO).
    :param args: (argparse.Namespace) arguments containing relevant model information.
    :param cent_obs_dim: (int or tuple) (centralized) observation dimension(s).
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """
    def __init__(self, 
            args, cent_obs_dim, device=torch.device("cpu")):
        super(R_Critic, self).__init__()
        self.hidden_size = args.hidden_size
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._recurrent_N = args.recurrent_N
        self._use_popart = args.use_popart
        self.tpdv = dict(dtype=torch.float32, device=device)
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][args.use_orthogonal]

        # cent_obs_dim = get_shape_from_obs_space(cent_obs_space)
        self.obs_encoder = MLPBase(args, cent_obs_dim)

        # # [LMC] Add dimension of context encoding
        # input_encoding_dim = self.hidden_size + context_dim

        # Add encoding layer
        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            self.rnn_encoder = RNNLayer(
                self.hidden_size, self.hidden_size, self._recurrent_N, 
                args.use_orthogonal)
        else:
            self.mlp_encoder = MLPNet(
                self.hidden_size, self.hidden_size, 1, 
                args.use_orthogonal, args.use_ReLU)

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))

        if self._use_popart:
            self.v_out = init_(PopArt(self.hidden_size, 1, device=device))
        else:
            self.v_out = init_(nn.Linear(self.hidden_size, 1))

        self.to(device)

    def forward(self, cent_obs, context, rnn_states, masks):
        """
        Compute actions from the given inputs.
        :param cent_obs: (np.ndarray / torch.Tensor) observation inputs into network.
        :param rnn_states: (np.ndarray / torch.Tensor) if RNN network, hidden states for RNN.
        :param masks: (np.ndarray / torch.Tensor) mask tensor denoting if RNN states should be reinitialized to zeros.

        :return values: (torch.Tensor) value function predictions.
        :return rnn_states: (torch.Tensor) updated RNN hidden states.
        """
        cent_obs = torch.from_numpy(cent_obs).to(**self.tpdv)
        rnn_states = torch.from_numpy(rnn_states).to(**self.tpdv)
        masks = torch.from_numpy(masks).to(**self.tpdv)

        critic_features = self.obs_encoder(cent_obs)

        # # [LMC] Concatenate incoming context encoding
        # critic_features = torch.cat((obs_enc, context), dim=-1)

        # Add encoding layer
        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            critic_features, rnn_states = self.rnn_encoder(
                critic_features, rnn_states, masks)
        else:
            critic_features = self.mlp_encoder(critic_features)

        values = self.v_out(critic_features)

        return values, rnn_states
