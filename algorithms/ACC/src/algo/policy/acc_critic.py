from torch import nn

from src.algo.nn_modules.mlp import MLPNetwork
from src.algo.nn_modules.rnn import RNNLayer
from src.algo.nn_modules.utils import init


class ACC_Critic(nn.Module):

    def __init__(self, args, shared_obs_dim):
        super(ACC_Critic, self).__init__()
        self.obs_encoder = MLPNetwork(
            shared_obs_dim, 
            args.hidden_dim, 
            args.hidden_dim, 
            args.policy_layer_N,
            out_activation_fn="relu")
        
        self.rnn_encoder = RNNLayer(
            args.hidden_dim, args.hidden_dim, args.policy_recurrent_N)

        def init_(m):
            return init(
                m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0))

        # Value head
        self.value_head = MLPNetwork(
            args.hidden_dim, 
            args.hidden_dim, 
            args.hidden_dim, 
            out_activation_fn="relu")
        self.act_v_out = init_(nn.Linear(args.hidden_dim, 1))
        self.comm_v_out = init_(nn.Linear(args.hidden_dim, 1))

        # Observation encoding head
        self.obs_enc_head = MLPNetwork(
            args.hidden_dim, args.context_dim, args.hidden_dim)

    def forward(self, shared_obs, rnn_states, masks, get_obs_encs=False):
        x = self.obs_encoder(shared_obs)

        x, new_rnn_states = self.rnn_encoder(x, rnn_states, masks)

        v = self.value_head(x)
        act_values, comm_values = self.act_v_out(v), self.comm_v_out(v)

        if get_obs_encs:
            obs_encs = self.obs_enc_head(x)
            return act_values, comm_values, new_rnn_states, obs_encs
        else:
            return act_values, comm_values, new_rnn_states