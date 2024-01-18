from torch import nn

from src.algo.nn_modules.mlp import MLPNetwork
from src.algo.nn_modules.rnn import RNNLayer


class Critic(nn.Module):

    def __init__(self, args, shared_obs_dim):
        super(Critic, self).__init__()
        self.obs_encoder = MLPNetwork(
            shared_obs_dim, 
            args.hidden_dim, 
            args.hidden_dim, 
            args.policy_layer_N)
        
        self.rnn_encoder = RNNLayer(
            args.hidden_dim, args.hidden_dim, args.policy_recurrent_N)

        gain = nn.init.calculate_gain(activation_fn)
        def init_(m):
            return init(
                m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), 
                gain=gain)

        self.v_out = init_(nn.Linear(args.hidden_dim, 1))

    def forward(self, shared_obs, rnn_states, masks):
        x = self.obs_encoder(obs)

        x, new_rnn_states = self.rnn_encoder(x, rnn_states, masks)

        values = self.v_out(x)

        return values, new_rnn_states