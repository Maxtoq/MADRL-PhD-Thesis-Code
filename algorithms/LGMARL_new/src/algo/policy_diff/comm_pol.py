import torch

from torch import nn

from src.algo.nn_modules.mlp import MLPNetwork
from src.algo.nn_modules.rnn import RNNLayer
from src.algo.nn_modules.distributions import DiagGaussian, Categorical


class CP_NoComm(nn.Module):

    def __init__(self, args):
        self.context_dim = args.context_dim

    def gen_comm(self, enc_obs, perfect_messages):
        comm_actions = torch.zeros(obs.shape[0], self.context_dim)
        comm_action_log_probs = torch.zeros(obs.shape[0], 1)
        comm_values = torch.zeros(obs.shape[0], 1)
        messages = None
        eval_comm_action_log_probs = None
        eval_comm_dist_entropy = None

        return messages, comm_actions, comm_action_log_probs, comm_values, \
            eval_comm_action_log_probs, eval_comm_dist_entropy

    def enc_comm(
            self, enc_obs, enc_joint_obs, messages, comm_rnn_states, masks):
        pol_input = enc_obs
        val_input = enc_joint_obs
        new_comm_rnn_states = torch.zeros_like(comm_rnn_states)

        return pol_input, val_input, new_comm_rnn_states

    # def set_device(self, device):
    #     pass


class CP_EmergentContinuous(nn.Module):

    def __init__(self, args):
        self.comm_in = MLPNetwork(
            args.hidden_dim, 
            args.context_dim, 
            args.hidden_dim, 
            out_activation_fn="tanh")

        self.comm_enc = RNNLayer(
            args.context_dim, 
            args.hidden_dim, 
            args.policy_recurrent_N)

    def gen_comm(self, enc_obs, perfect_messages):
        comm_actions = self.comm_in(enc_obs)
        messages = comm_actions.clone()

        eval_comm_action_log_probs = None
        eval_comm_dist_entropy = None
        comm_action_log_probs = torch.zeros(enc_obs.shape[0], 1)
        comm_values = torch.zeros(enc_obs.shape[0], 1)

        return messages, comm_actions, comm_action_log_probs, comm_values, \
            eval_comm_action_log_probs, eval_comm_dist_entropy

    def enc_comm(
            self, enc_obs, enc_joint_obs, messages, comm_rnn_states, masks):
        comm_enc, new_comm_rnn_states = self.comm_encoder(
                messages, comm_rnn_states, masks)

        pol_input = torch.concatenate((comm_enc, enc_obs), dim=-1)
        val_input = torch.concatenate((comm_enc, enc_joint_obs), dim=-1)

        return pol_input, val_input, new_comm_rnn_states

    # def set_device(self, device):
    #     pass


class CP_Perfect(nn.Module):

    def __init__(self, args):
        self.comm_in = MLPNetwork(
            args.hidden_dim, 
            args.context_dim, 
            args.hidden_dim, 
            out_activation_fn="tanh")

        self.comm_enc = RNNLayer(
            args.context_dim, 
            args.hidden_dim, 
            args.policy_recurrent_N)

    def gen_comm(self, enc_obs, perfect_messages):
        comm_actions = self.comm_in(enc_obs)
        messages = perfect_messages

        eval_comm_action_log_probs = None
        eval_comm_dist_entropy = None
        comm_action_log_probs = torch.zeros(enc_obs.shape[0], 1)
        comm_values = torch.zeros(enc_obs.shape[0], 1)

        return messages, comm_actions, comm_action_log_probs, comm_values, \
            eval_comm_action_log_probs, eval_comm_dist_entropy

    def enc_comm(
            self, enc_obs, enc_joint_obs, messages, comm_rnn_states, masks):
        comm_enc, new_comm_rnn_states = self.comm_encoder(
                messages, comm_rnn_states, masks)

        pol_input = torch.concatenate((comm_enc, enc_obs), dim=-1)
        val_input = torch.concatenate((comm_enc, enc_joint_obs), dim=-1)

        return pol_input, val_input, new_comm_rnn_states

    # def set_device(self, device):
    #     pass


class CP_EmergentDiscrete_Diff(nn.Module):

    def __init__(self, args):
        # self.lang_learner = lang_learner
        self.device = device

        self.comm_in = MLPNetwork(
            args.hidden_dim, 
            args.context_dim, 
            args.hidden_dim, 
            out_activation_fn="tanh")

        self.comm_enc = RNNLayer(
            args.context_dim, 
            args.hidden_dim, 
            args.policy_recurrent_N)

    def gen_comm(self, enc_obs, perfect_messages):
        comm_actions = self.comm_pol(enc_obs)

        messages = None

        comm_action_log_probs = torch.zeros(enc_obs.shape[0], 1)
        comm_values = torch.zeros(enc_obs.shape[0], 1)
        eval_comm_action_log_probs = None
        eval_comm_dist_entropy = None

        return messages, comm_actions, comm_action_log_probs, comm_values, \
            eval_comm_action_log_probs, eval_comm_dist_entropy

    def enc_comm(
            self, enc_obs, enc_joint_obs, messages, comm_rnn_states, masks):
        comm_enc, new_comm_rnn_states = self.comm_encoder(
                messages, comm_rnn_states, masks)

        pol_input = torch.concatenate((comm_enc, enc_obs), dim=-1)
        val_input = torch.concatenate((comm_enc, enc_joint_obs), dim=-1)

        return pol_input, val_input, new_comm_rnn_states

    # def set_device(self, device):
    #     self.device = device