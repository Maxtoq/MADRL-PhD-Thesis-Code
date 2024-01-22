from torch import nn

from src.algo.nn_modules.mlp import MLPNetwork
from src.algo.nn_modules.rnn import RNNLayer
from src.algo.nn_modules.distributions import DiagGaussian, Categorical


class ActorCommunicator(nn.Module):

    def __init__(self, args, obs_dim, act_dim):
        super(ActorCommunicator, self).__init__()
        self.obs_encoder = MLPNetwork(
            obs_dim, args.hidden_dim, args.hidden_dim, args.policy_layer_N,
            out_activation_fn="relu")
        
        self.rnn_encoder = RNNLayer(
            args.hidden_dim, args.hidden_dim, args.policy_recurrent_N)

        self.action_head = Categorical(args.hidden_dim, act_dim)

        self.comm_head = DiagGaussian(args.hidden_dim, args.context_dim)

    def forward(self, obs, rnn_states, masks):
        x = self.obs_encoder(obs)

        x, new_rnn_states = self.rnn_encoder(x, rnn_states, masks)

        # Get env actions
        env_action_logits = self.action_head(x)
        env_actions = env_action_logits.sample() 
        env_action_log_probs = env_action_logits.log_probs(env_actions)

        # Get comm actions
        comm_action_logits = self.comm_head(x)
        comm_actions = comm_action_logits.sample() 
        comm_action_log_probs = comm_action_logits.log_probs(comm_actions)

        return env_actions, env_action_log_probs, comm_actions, \
                comm_action_log_probs, new_rnn_states

    def evaluate_actions(self, 
            obs, rnn_states, env_actions, comm_actions, masks, eval_comm=True):
        """
        Compute log probability and entropy of given actions.
        :param obs: (torch.Tensor) observation inputs into network.
        :param env_actions: (torch.Tensor) environment actions whose log 
            probabilites and entropy to compute.
        :param comm_actions: (torch.Tensor) communication actions whose log 
            probabilites and entropy to compute.
        :param rnn_states: (torch.Tensor) if RNN network, hidden states for RNN.
        :param masks: (torch.Tensor) mask tensor denoting if hidden states 
            should be reinitialized to zeros.
        :param eval_comm: (bool) whether to compute comm_actions probs.

        :return env_action_log_probs: (torch.Tensor) log probabilities of the
            environment actions.
        :return env_dist_entropy: (torch.Tensor) environment action 
            distribution entropy for the given inputs.
        :return comm_dist_entropy: (torch.Tensor) communication action 
            distribution entropy for the given inputs.
        :return comm_action_log_probs: (torch.Tensor) log probabilities of the
            communication actions.
        """
        x = self.obs_encoder(obs)

        x, new_rnn_states = self.rnn_encoder(x, rnn_states, masks)

        # Eval environment actions
        env_action_logits = self.action_head(x)
        env_action_log_probs = env_action_logits.log_probs(env_actions)
        env_dist_entropy = env_action_logits.entropy().mean()

        # Eval communication actions
        if eval_comm:
            comm_action_logits = self.comm_head(x)
            comm_action_log_probs = comm_action_logits.log_probs(env_actions)
            comm_dist_entropy = comm_action_logits.entropy().mean()
        else:
            comm_action_log_probs = None
            comm_dist_entropy = None

        return env_action_log_probs, env_dist_entropy, comm_action_log_probs, \
                comm_dist_entropy
        