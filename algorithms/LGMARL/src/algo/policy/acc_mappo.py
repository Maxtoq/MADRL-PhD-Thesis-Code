import torch
import numpy as np

from .acc_agent import ACC_Agent
from .acc_mappo_trainer import ACC_MAPPOTrainAlgo
from .utils import torch2numpy


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
            self.agents = [ACC_Agent(args, obs_dim, shared_obs_dim, act_dim)]
        else:
            self.agents = [
                ACC_Agent(args, obs_dim, shared_obs_dim, act_dim)
                for a_i in range(self.n_agents)]

        self.trainer = ACC_MAPPOTrainAlgo(args, self.agents, device)

    def prep_rollout(self, device=None):
        self.device = self.train_device if device is None else device
        for a in self.agents:
            a.act_comm.eval()
            a.act_comm.to(self.device)
            a.critic.eval()
            a.critic.to(self.device)
        self.trainer.device = self.device
        if self.trainer.value_normalizer is not None:
            self.trainer.value_normalizer.to(self.device)

    def prep_training(self, device=None):
        if device is not None:
            self.train_device = device
        self.device = self.train_device
        for a in self.agents:
            a.act_comm.train()
            a.act_comm.to(self.train_device)
            a.critic.train()
            a.critic.to(self.train_device)
        self.trainer.device = self.train_device
        if self.trainer.value_normalizer is not None:
            self.trainer.value_normalizer.to(self.train_device)

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

            values, actions, action_log_probs, comm_actions, comm_action_log_probs, \
                rnn_states, critic_rnn_states = self.agents[0](
                    obs, shared_obs, rnn_states, critic_rnn_states, masks)

            values = torch2numpy(
                values.reshape(self.n_envs, self.n_agents, -1))
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
            values = []
            actions = []
            action_log_probs = []
            comm_actions = []
            comm_action_log_probs = []
            new_rnn_states = []
            new_critic_rnn_states = []
            for a_i in range(self.n_agents):
                value, action, action_log_prob, comm_action, \
                comm_action_log_prob, new_rnn_state, \
                    new_critic_rnn_state = self.agents[a_i](
                        obs[:, a_i], 
                        shared_obs[:, a_i], 
                        rnn_states[:, a_i], 
                        critic_rnn_states[:, a_i], 
                        masks[:, a_i])

                values.append(value)
                actions.append(action)
                action_log_probs.append(action_log_prob)
                comm_actions.append(comm_action)
                comm_action_log_probs.append(comm_action_log_prob)
                new_rnn_states.append(new_rnn_state)
                new_critic_rnn_states.append(new_critic_rnn_state)

            values = torch2numpy(torch.stack(values, dim=1))
            actions = torch2numpy(torch.stack(actions, dim=1))
            action_log_probs = torch2numpy(torch.stack(action_log_probs, dim=1))
            comm_actions = torch2numpy(torch.stack(comm_actions, dim=1))
            comm_action_log_probs = torch2numpy(
                torch.stack(comm_action_log_probs, dim=1))
            rnn_states = torch2numpy(torch.stack(new_rnn_states, dim=1))
            critic_rnn_states = torch2numpy(
                torch.stack(new_critic_rnn_states, dim=1))

        return values, actions, action_log_probs, comm_actions, \
                comm_action_log_probs, rnn_states, critic_rnn_states

    @torch.no_grad()
    def compute_last_value(self, shared_obs, critic_rnn_states, masks):
        shared_obs = torch.from_numpy(shared_obs).to(self.device)
        critic_rnn_states = torch.from_numpy(critic_rnn_states).to(self.device)
        masks = torch.from_numpy(masks).to(self.device)

        if self.share_params:
            next_values = self.agents[0].get_values(
                shared_obs.reshape(self.n_envs * self.n_agents, -1),
                critic_rnn_states.reshape(
                    self.n_envs * self.n_agents, self.recurrent_N, -1),
                masks.reshape(self.n_envs * self.n_agents, -1))

            next_values = torch2numpy(
                next_values.reshape(self.n_envs, self.n_agents, -1))
        else:
            next_values = []
            for a_i in range(self.n_agents):
                next_values.append(self.agents[a_i].get_values(
                    shared_obs[:, a_i], critic_rnn_states[:, a_i], masks[:, a_i]))
                
            next_values = torch2numpy(torch.stack(next_values, dim=1))

        return next_values

    # def train(self, warmup=False, train_comm_head=True):
    #     # Compute last value
    #     self._compute_last_value()
    #     # Train
    #     self.prep_training()
    #     for a in self.agents:
    #         a.warmup_lr(warmup)
    #     losses = self.trainer.train(self.buffer, train_comm_head)
    #     return losses

    def get_save_dict(self):
        self.prep_rollout("cpu")
        save_dict = {
            "act_comms": [a.act_comm.state_dict() for a in self.agents],
            "critics": [a.critic.state_dict() for a in self.agents],
            # TODO Move ça dans lgmarl
            "vnorm": self.trainer.value_normalizer.state_dict()
        }
        return save_dict

    def load_params(self, params):
        for a, ac, c in zip(self.agents, params["act_comms"], params["critics"]):
            a.act_comm.load_state_dict(ac)
            a.critic.load_state_dict(c)
        self.trainer.value_normalizer.load_state_dict(params["vnorm"])

