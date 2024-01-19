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
        action_logits = self.action_head(x)
        actions = action_logits.sample() 
        action_log_probs = action_logits.log_probs(actions)

        # Get comm actions
        comm_action_logits = self.comm_head(x)
        comm_actions = comm_action_logits.sample() 
        comm_action_log_probs = comm_action_logits.log_probs(comm_actions)

        return actions, action_log_probs, comm_actions, comm_action_log_probs, \
                new_rnn_states
        