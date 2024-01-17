from src.algo.nn_modules.mlp import MLPNetwork
from src.algo.nn_modules.rnn import RNNLayer
from src.algo.nn_modules.ditributions import DiagGaussian, Categorical


class ActorCommunicator(nn.Module):

    def __init__(self, args, obs_dim, act_space):
        super(ActorCommunicator, self).__init__()

        self.obs_encoder = MLPNetwork(
            obs_dim, args.hidden_dim, args.hidden_dim, args.policy_layer_N)
        
        self.rnn_encoder = RNNLayer(
            args.hidden_dim, args.hidden_dim, args.policy_recurrent_N)

        action_dim = act_space[1].n
        self.action_head = Categorical(args.hidden_dim, action_dim)

        comm_context_dim = args.context_dim
        self.comm_head = DiagGaussian(args.hidden_dim, comm_context_dim)

    def forward(self, obs, rnn_states, masks):
        x = self.obs_encoder(obs)

        x, new_rnn_states = self.rnn_encoder(x, rnn_states, masks)

        # Get actions
        action_logits = self.action_out(x)
        actions = action_logits.sample() 
        action_log_probs = action_logits.log_probs(actions)


        