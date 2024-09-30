import torch

from torch import nn

from src.algo.nn_modules.mlp import MLPNetwork
from src.algo.nn_modules.rnn import RNNLayer
from src.algo.nn_modules.distributions import DiagGaussian, Categorical


class CommPolicy(nn.Module):

    def __init__(self, args, n_agents, obs_dim):
        super(CommPolicy, self).__init__()
        self.context_dim = args.context_dim
        self.comm_type = args.comm_type
        self.n_agents = n_agents

        self.comm_in = MLPNetwork(
            args.hidden_dim, 
            args.context_dim, 
            args.hidden_dim, 
            out_activation_fn="tanh")

        if self.comm_type == "emergent_continuous":
            in_comm_enc = n_agents * args.context_dim
        elif self.comm_type == "obs":
            in_comm_enc = n_agents * obs_dim
        else:
            in_comm_enc = args.context_dim
        self.comm_encoder = RNNLayer(
            in_comm_enc, 
            args.hidden_dim, 
            args.policy_recurrent_N)

    def gen_comm(self, enc_obs, perfect_messages, obs):
        if self.comm_type == "no_comm":
            comm_actions = torch.zeros(enc_obs.shape[0], self.context_dim)
            comm_action_log_probs = torch.zeros(enc_obs.shape[0], 1)
            comm_values = torch.zeros(enc_obs.shape[0], 1)
            messages = None
            eval_comm_action_log_probs = None
            eval_comm_dist_entropy = None

        elif self.comm_type == "emergent_continuous":
            comm_actions = self.comm_in(enc_obs)
            messages = comm_actions.clone() 

            eval_comm_action_log_probs = None
            eval_comm_dist_entropy = None
            comm_action_log_probs = torch.zeros(enc_obs.shape[0], 1)
            comm_values = torch.zeros(enc_obs.shape[0], 1)

        elif self.comm_type == "emergent_discrete_lang":
            comm_actions = self.comm_in(enc_obs)

            messages = None
            comm_action_log_probs = torch.zeros(enc_obs.shape[0], 1)
            comm_values = torch.zeros(enc_obs.shape[0], 1)
            eval_comm_action_log_probs = None
            eval_comm_dist_entropy = None
        
        elif self.comm_type in ["perfect", "language_sup", "perfect+no_lang"]:
            comm_actions = self.comm_in(enc_obs)
            messages = perfect_messages

            eval_comm_action_log_probs = None
            eval_comm_dist_entropy = None
            comm_action_log_probs = torch.zeros(enc_obs.shape[0], 1)
            comm_values = torch.zeros(enc_obs.shape[0], 1)
        
        elif self.comm_type == "obs":
            messages = obs

            comm_actions = torch.zeros(enc_obs.shape[0], self.context_dim)
            comm_action_log_probs = torch.zeros(enc_obs.shape[0], 1)
            comm_values = torch.zeros(enc_obs.shape[0], 1)
            eval_comm_action_log_probs = None
            eval_comm_dist_entropy = None

        elif self.comm_type == "no_comm+lang":
            comm_actions = self.comm_in(enc_obs)

            comm_action_log_probs = torch.zeros(enc_obs.shape[0], 1)
            comm_values = torch.zeros(enc_obs.shape[0], 1)
            messages = None
            eval_comm_action_log_probs = None
            eval_comm_dist_entropy = None

        return messages, comm_actions, comm_action_log_probs, comm_values, \
            eval_comm_action_log_probs, eval_comm_dist_entropy

    def enc_comm(
            self, enc_obs, enc_joint_obs, messages, comm_rnn_states, masks):
        if "no_comm" in self.comm_type:
            pol_input = enc_obs
            val_input = enc_joint_obs
            new_comm_rnn_states = torch.zeros_like(comm_rnn_states)
        
        else:
        # elif self.comm_type in [
        #         "emergent_continuous", "emergent_discrete_lang", "perfect", 
        #         "language_sup"]:
            comm_enc, new_comm_rnn_states = self.comm_encoder(
                messages, comm_rnn_states, masks)

            pol_input = torch.concatenate((comm_enc, enc_obs), dim=-1)
            val_input = torch.concatenate((comm_enc, enc_joint_obs), dim=-1)

        return pol_input, val_input, new_comm_rnn_states

