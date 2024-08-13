import torch
import numpy as np

from torch import nn

from .comm_agent import Comm_Agent
from .utils import torch2numpy, update_lr


class Comm_MAPPO():

    def __init__(self, args, lang_learner, n_agents, obs_dim, 
                 shared_obs_dim, act_dim, device):
        self.args = args
        self.lang_learner = lang_learner
        self.n_agents = n_agents
        self.device = device
        # self.context_dim = args.context_dim
        # self.n_envs = args.n_parallel_envs
        # self.recurrent_N = args.policy_recurrent_N
        # self.hidden_dim = args.hidden_dim
        # self.lr = args.lr
        self.share_params = args.share_params
        self.comm_type = args.comm_type

        if self.share_params:
            raise NotImplementedError
            self.agents = [
                Comm_Agent(args, obs_dim, shared_obs_dim, act_dim)]
        else:
            self.agents = [
                Comm_Agent(args, obs_dim, shared_obs_dim, act_dim)
                for a_i in range(self.n_agents)]

        self.eval = False

    @torch.no_grad()
    def compute_last_value(self, joint_obs, joint_obs_rnn_states, masks):
        # if self.share_params:
        #     next_act_values, next_comm_values = self.agents[0].get_values(
        #         shared_obs.reshape(self.n_envs * self.n_agents, -1),
        #         critic_rnn_states.reshape(
        #             self.n_envs * self.n_agents, self.recurrent_N, -1),
        #         masks.reshape(self.n_envs * self.n_agents, -1))

        #     next_act_values = torch2numpy(
        #         next_act_values.reshape(self.n_envs, self.n_agents, -1))
        #     next_comm_values = torch2numpy(
        #         next_comm_values.reshape(self.n_envs, self.n_agents, -1))
        # else:
        next_act_values = []
        next_comm_values = []
        for a_i in range(self.n_agents):
            next_act_value, next_comm_value = self.agents[a_i].get_values(
                joint_obs[:, a_i], joint_obs_rnn_states[:, a_i], masks[:, a_i])
            next_act_values.append(next_act_value)
            next_comm_values.append(next_comm_value)
            
        next_act_values = torch2numpy(torch.stack(next_act_values, dim=1))
        next_comm_values = torch2numpy(torch.stack(next_comm_values, dim=1))

        return next_act_values, next_comm_values

    def prep_rollout(self, device=None):
        if device is not None:
            self.device = device
        for a in self.agents:
            a.eval()
            a.to(self.device)

    def prep_training(self, device=None):
        if device is not None:
            self.device = device
        for a in self.agents:
            a.train()
            a.to(self.device)

    def get_actions(
            self, obs, joint_obs, obs_rnn_states, joint_obs_rnn_states, 
            comm_rnn_states, masks, deterministic=False):
        obs = torch.from_numpy(obs).to(self.device)
        joint_obs = torch.from_numpy(joint_obs).to(self.device)
        obs_rnn_states = torch.from_numpy(obs_rnn_states).to(self.device)
        joint_obs_rnn_states = torch.from_numpy(
            joint_obs_rnn_states).to(self.device)
        comm_rnn_states = torch.from_numpy(comm_rnn_states).to(self.device)
        masks = torch.from_numpy(masks).to(self.device)

        # TODO: handle perfect messages

        # Generate comm
        agents_messages = []
        agents_enc_obs = []
        agents_enc_joint_obs = []
        agents_comm_actions = []
        agents_comm_action_log_probs = []
        agents_comm_values = []
        agents_new_obs_rnn_states = []
        agents_new_joint_obs_rnn_states = []
        for a_i in range(self.n_agents):
            messages, enc_obs, enc_joint_obs, comm_actions, \
                comm_action_log_probs, comm_values, new_obs_rnn_states, \
                new_joint_obs_rnn_states \
                = self.agents[a_i].get_message(
                    obs[:, a_i], 
                    joint_obs[:, a_i], 
                    obs_rnn_states[:, a_i], 
                    joint_obs_rnn_states[:, a_i], 
                    masks[:, a_i], 
                    deterministic)

            agents_messages.append(messages)
            agents_enc_obs.append(enc_obs)
            agents_enc_joint_obs.append(enc_joint_obs)
            agents_comm_actions.append(comm_actions)
            agents_comm_action_log_probs.append(comm_action_log_probs)
            agents_comm_values.append(comm_values)
            agents_new_obs_rnn_states.append(new_obs_rnn_states)
            agents_new_joint_obs_rnn_states.append(new_joint_obs_rnn_states)

        # Aggregate messages
        if self.comm_type == "no_comm":
            messages = None

        # Generate actions
        agents_actions = []
        agents_action_log_probs = []
        agents_values = []
        agents_new_comm_rnn_states = []
        for a_i in range(self.n_agents):
            actions, action_log_probs, values, new_comm_rnn_states = \
                self.agents[a_i].get_actions(
                    messages, 
                    agents_enc_obs[a_i],
                    agents_enc_joint_obs[a_i],
                    comm_rnn_states[:, a_i],
                    masks[:, a_i],
                    deterministic)

            agents_actions.append(actions)
            agents_action_log_probs.append(action_log_probs)
            agents_values.append(values)
            agents_new_comm_rnn_states.append(new_comm_rnn_states)

        actions = torch2numpy(torch.stack(agents_actions, dim=1))
        action_log_probs = torch2numpy(
            torch.stack(agents_action_log_probs, dim=1))
        values = torch2numpy(torch.stack(agents_values, dim=1))
        comm_actions = torch2numpy(torch.stack(agents_comm_actions, dim=1))
        comm_action_log_probs = torch2numpy(
            torch.stack(agents_comm_action_log_probs, dim=1))
        comm_values = torch2numpy(torch.stack(agents_comm_values, dim=1))
        new_obs_rnn_states = torch2numpy(
            torch.stack(agents_new_obs_rnn_states, dim=1))
        new_joint_obs_rnn_states = torch2numpy(
            torch.stack(agents_new_joint_obs_rnn_states, dim=1))
        new_comm_rnn_states = torch2numpy(
            torch.stack(agents_new_comm_rnn_states, dim=1))

        return actions, action_log_probs, values, comm_actions, \
            comm_action_log_probs, comm_values, new_obs_rnn_states, \
            new_joint_obs_rnn_states, new_comm_rnn_states

    def get_save_dict(self):
        self.prep_rollout("cpu")
        save_dict = {
            "agents": [a.state_dict() for a in self.agents]}
        return save_dict

    def load_params(self, params):
        for a, ap in zip(self.agents, params["agents"]):
            a.load_state_dict(ap)
