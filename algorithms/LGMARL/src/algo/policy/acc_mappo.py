import torch
import numpy as np

from .acc_agent import ACC_Agent
from .utils import torch2numpy, update_lr


class ACC_MAPPO:

    def __init__(self, args, lang_learner, n_agents, obs_dim, 
                 shared_obs_dim, act_dim, device):
        self.args = args
        self.lang_learner = lang_learner
        self.n_agents = n_agents
        self.train_device = device
        self.context_dim = args.context_dim
        self.n_envs = args.n_parallel_envs
        self.recurrent_N = args.policy_recurrent_N
        self.hidden_dim = args.hidden_dim
        self.lr = args.lr
        self.share_params = args.share_params

        self.device = self.train_device

        if self.share_params:
            self.agents = [
                ACC_Agent(args, obs_dim, shared_obs_dim, act_dim)] # , lang_learner)]
        else:
            self.agents = [
                ACC_Agent(args, obs_dim, shared_obs_dim, act_dim) # , lang_learner)
                for a_i in range(self.n_agents)]

        self.eval = False

    def prep_rollout(self, device=None):
        self.device = self.train_device if device is None else device
        for a in self.agents:
            a.act_comm.eval()
            a.act_comm.to(self.device)
            a.critic.eval()
            a.critic.to(self.device)

    def prep_training(self, device=None):
        if device is not None:
            self.train_device = device
        self.device = self.train_device
        for a in self.agents:
            a.act_comm.train()
            a.act_comm.to(self.train_device)
            a.critic.train()
            a.critic.to(self.train_device)

    # def update_lrs(self, new_capt_lr):
    #     for agent in self.agents:
    #         break
    #         update_lr(agent.capt_optim, new_capt_lr)

    @torch.no_grad()
    def get_actions(self, obs, shared_obs, rnn_states, critic_rnn_states, masks):
        obs = torch.from_numpy(obs).to(self.device)
        shared_obs = torch.from_numpy(shared_obs).to(self.device)
        rnn_states = torch.from_numpy(rnn_states).to(self.device)
        critic_rnn_states = torch.from_numpy(
            critic_rnn_states).to(self.device)
        masks = torch.from_numpy(masks).to(self.device)

        if self.share_params:
            obs = obs.reshape(self.n_envs * self.n_agents, -1)
            shared_obs = shared_obs.reshape(self.n_envs * self.n_agents, -1)
            rnn_states = rnn_states.reshape(
                self.n_envs * self.n_agents, self.recurrent_N, -1)
            critic_rnn_states = critic_rnn_states.reshape(
                self.n_envs * self.n_agents, self.recurrent_N, -1)
            masks = masks.reshape(self.n_envs * self.n_agents, -1)

            act_values, comm_values, actions, action_log_probs, comm_actions, \
                comm_action_log_probs, rnn_states, critic_rnn_states \
                    = self.agents[0](
                        obs, 
                        shared_obs, 
                        rnn_states, 
                        critic_rnn_states, 
                        masks, 
                        self.eval)

            act_values = torch2numpy(
                act_values.reshape(self.n_envs, self.n_agents, -1))
            comm_values = torch2numpy(
                comm_values.reshape(self.n_envs, self.n_agents, -1))
            actions = torch2numpy(
                actions.reshape(self.n_envs, self.n_agents, -1))
            action_log_probs = torch2numpy(action_log_probs.reshape(
                self.n_envs, self.n_agents, -1))
            comm_actions = torch2numpy(
                comm_actions.reshape(self.n_envs, self.n_agents, -1))
            comm_action_log_probs = torch2numpy(comm_action_log_probs.reshape(
                self.n_envs, self.n_agents, -1))
            rnn_states = torch2numpy(rnn_states.reshape(
                self.n_envs, self.n_agents, self.recurrent_N, -1))
            critic_rnn_states = torch2numpy(critic_rnn_states.reshape(
                self.n_envs, self.n_agents, self.recurrent_N, -1))
        else:
            act_values = []
            comm_values = []
            actions = []
            action_log_probs = []
            comm_actions = []
            comm_action_log_probs = []
            new_rnn_states = []
            new_critic_rnn_states = []
            for a_i in range(self.n_agents):
                act_value, comm_value, action, action_log_prob, comm_action, \
                    comm_action_log_prob, new_rnn_state, new_critic_rnn_state \
                        = self.agents[a_i](
                            obs[:, a_i], 
                            shared_obs[:, a_i], 
                            rnn_states[:, a_i], 
                            critic_rnn_states[:, a_i], 
                            masks[:, a_i], 
                            self.eval)

                act_values.append(act_value)
                comm_values.append(comm_value)
                actions.append(action)
                action_log_probs.append(action_log_prob)
                comm_actions.append(comm_action)
                comm_action_log_probs.append(comm_action_log_prob)
                new_rnn_states.append(new_rnn_state)
                new_critic_rnn_states.append(new_critic_rnn_state)

            act_values = torch2numpy(torch.stack(act_values, dim=1))
            comm_values = torch2numpy(torch.stack(comm_values, dim=1))
            actions = torch2numpy(torch.stack(actions, dim=1))
            action_log_probs = torch2numpy(torch.stack(action_log_probs, dim=1))
            comm_actions = torch2numpy(torch.stack(comm_actions, dim=1))
            comm_action_log_probs = torch2numpy(
                torch.stack(comm_action_log_probs, dim=1))
            rnn_states = torch2numpy(torch.stack(new_rnn_states, dim=1))
            critic_rnn_states = torch2numpy(
                torch.stack(new_critic_rnn_states, dim=1))

        return act_values, comm_values, actions, action_log_probs, comm_actions, \
                comm_action_log_probs, rnn_states, critic_rnn_states

    @torch.no_grad()
    def compute_last_value(self, shared_obs, critic_rnn_states, masks):
        shared_obs = torch.from_numpy(shared_obs).to(self.device)
        critic_rnn_states = torch.from_numpy(critic_rnn_states).to(self.device)
        masks = torch.from_numpy(masks).to(self.device)

        if self.share_params:
            next_act_values, next_comm_values = self.agents[0].get_values(
                shared_obs.reshape(self.n_envs * self.n_agents, -1),
                critic_rnn_states.reshape(
                    self.n_envs * self.n_agents, self.recurrent_N, -1),
                masks.reshape(self.n_envs * self.n_agents, -1))

            next_act_values = torch2numpy(
                next_act_values.reshape(self.n_envs, self.n_agents, -1))
            next_comm_values = torch2numpy(
                next_comm_values.reshape(self.n_envs, self.n_agents, -1))
        else:
            next_act_values = []
            next_comm_values = []
            for a_i in range(self.n_agents):
                next_act_value, next_comm_value = self.agents[a_i].get_values(
                    shared_obs[:, a_i], critic_rnn_states[:, a_i], masks[:, a_i])
                next_act_values.append(next_act_value)
                next_comm_values.append(next_comm_value)
                
            next_act_values = torch2numpy(torch.stack(next_act_values, dim=1))
            next_comm_values = torch2numpy(torch.stack(next_comm_values, dim=1))

        return next_act_values, next_comm_values

    def get_save_dict(self):
        self.prep_rollout("cpu")
        save_dict = {
            "act_comms": [a.act_comm.state_dict() for a in self.agents],
            "critics": [a.critic.state_dict() for a in self.agents]}
        return save_dict

    def load_params(self, params):
        for a, ac, c in zip(self.agents, params["act_comms"], params["critics"]):
            a.act_comm.load_state_dict(ac)
            a.critic.load_state_dict(c)

