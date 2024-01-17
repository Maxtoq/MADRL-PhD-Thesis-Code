from src.algo.nn_modules.mlp import MLPNetwork
from src.algo.nn_modules.rnn import RNNLayer


class ActorCommunicator(nn.Module):

    def __init__(self, args, obs_dim, act_space, device):
        super(ActorCommunicator, self).__init__()

        # Input encoder
        self.obs_encoder = MLPNetwork(
            obs_dim, args.hidden_dim, args.hidden_dim, args.policy_layer_N)
        
        self.rnn_encoder = RNNLayer(
            args.hidden_dim, args.hidden_dim, args.policy_recurrent_N)

        action_dim = act_space[1].n
        self.action_head 
        comm_context_dim = args.context_dim
        