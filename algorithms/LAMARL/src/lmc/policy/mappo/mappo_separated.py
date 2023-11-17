import torch
import numpy as np

from .buffer_separated import SeparatedReplayBuffer
from .policy import R_MAPPOPolicy
from .train_algo import R_MAPPOTrainAlgo

def torch2numpy(x):
    return x.detach().cpu().numpy()


class MAPPO:
    """
    Class handling training of MAPPO from paper "The Surprising Effectiveness 
    of PPO in Cooperative Multi-Agent Games" (https://arxiv.org/abs/2103.01955),
    with different parameter sets for each agent.
    :param args: (dict) all arguments for training
    :param n_agents: (int) number of agents
    :param obs_dim: (int) observation dimensions, for each agent
    :param shared_obs_dim: (int) centralized observation dimensions, for 
        each agent
    :param act_space: (gym.Space) action dimensions, for each agent
    :param device: (torch.device) cuda device used for training
    """
    def __init__(self, 
            args, n_agents, obs_dim, shared_obs_dim, act_space, device):
        self.args = args
        self.n_agents = n_agents
        self.use_centralized_V = self.args["use_centralized_V"]
        self.train_device = device

        # Set variant
        if self.args["policy_algo"] == "rmappo":
            self.args["use_recurrent_policy"] = True
            self.args["use_naive_recurrent_policy"] = False
        elif self.args["policy_algo"] == "mappo":
            self.args["use_recurrent_policy"] = False 
            self.args["use_naive_recurrent_policy"] = False
        elif self.args["policy_algo"] == "ippo":
            self.use_centralized_V = False
        else:
            raise NotImplementedError("Bad param given for policy_algo.")

        # Init agent policies, train algo and buffer
        self.policy = []
        self.trainer = []
        self.buffer = []
        for a_i in range(self.n_agents):
            if self.use_centralized_V:
                sod = shared_obs_dim
            else:
                sod = obs_dim
            # Policy network
            po = R_MAPPOPolicy(
                self.args,
                obs_dim,
                sod,
                act_space,
                device=device)
            self.policy.append(po)

            # Algorithm
            tr = R_MAPPOTrainAlgo(
                self.args, po, device=device)
            self.trainer.append(tr)

            # Buffer
            bu = SeparatedReplayBuffer(
                self.args, 
                obs_dim, 
                sod,
                act_space)
            self.buffer.append(bu)

    def prep_rollout(self, device=None):
        if device is None:
            device = self.train_device
        for tr in self.trainer:
            tr.prep_rollout(device)

    def prep_training(self):
        for tr in self.trainer:
            tr.prep_training(self.train_device)

    def reset_buffer(self):
        # """
        # Initialize the buffer with first observations.
        # :param obs: (numpy.ndarray) first observations
        # """
        for a_i in range(self.n_agents):
            self.buffer[a_i].reset_episode()
            # if not self.use_centralized_V:
            #     shared_obs = np.array(list(obs[:, a_i]))
            # self.buffer[a_i].shared_obs[0] = shared_obs.copy()
            # self.buffer[a_i].obs[0] = np.array(list(obs[:, a_i])).copy()

    @torch.no_grad()
    def get_actions(self):
        values = []
        actions = []
        temp_actions_env = []
        action_log_probs = []
        rnn_states = []
        rnn_states_critic = []

        for a_i in range(self.n_agents):
            value, action, action_log_prob, rnn_state, rnn_state_critic \
                = self.trainer[a_i].policy.get_actions(
                    *self.buffer[a_i].get_act_params())
            # [agents, envs, dim]
            values.append(torch2numpy(value))
            action = torch2numpy(action)            

            actions.append(action)
            action_log_probs.append(torch2numpy(action_log_prob))
            rnn_states.append(torch2numpy(rnn_state))
            rnn_states_critic.append(torch2numpy(rnn_state_critic))

        values = np.array(values).transpose(1, 0, 2)
        actions = np.array(actions).transpose(1, 0, 2)
        action_log_probs = np.array(action_log_probs).transpose(1, 0, 2)
        rnn_states = np.array(rnn_states).transpose(1, 0, 2, 3)
        rnn_states_critic = np.array(rnn_states_critic).transpose(1, 0, 2, 3)

        return values, actions, action_log_probs, rnn_states, \
               rnn_states_critic

    def store_obs(self, obs, shared_obs):
        for a_i in range(self.n_agents):
            if not self.use_centralized_V:
                shared_obs = np.array(list(obs[:, a_i]))
            self.buffer[a_i].insert_obs(
                np.array(list(obs[:, a_i])).copy(),
                shared_obs.copy())

    def store_act(self, rewards, dones, infos, values, actions, 
            action_log_probs, rnn_states, rnn_states_critic):
        rnn_states[dones == True] = np.zeros(
            ((dones == True).sum(), 
             self.args["recurrent_N"], 
             self.args["hidden_size"]),
            dtype=np.float32)
        rnn_states_critic[dones == True] = np.zeros(
            ((dones == True).sum(), 
             self.args["recurrent_N"], 
             self.args["hidden_size"]),
            dtype=np.float32)
        masks = np.ones(
            (self.args["n_parallel_envs"], self.n_agents, 1), dtype=np.float32)
        masks[dones == True] = np.zeros(
            ((dones == True).sum(), 1), dtype=np.float32)

        for a_i in range(self.n_agents):
            # if not self.use_centralized_V:
            #     shared_obs = np.array(list(obs[:, a_i]))
            self.buffer[a_i].insert_act(
                rnn_states[:, a_i],
                rnn_states_critic[:, a_i],
                actions[:, a_i],
                action_log_probs[:, a_i],
                values[:, a_i],
                rewards[:, a_i],
                masks[:, a_i])

    @torch.no_grad()
    def compute_last_value(self):
        for a_i in range(self.n_agents):
            next_value = self.trainer[a_i].policy.get_values(
                self.buffer[a_i].shared_obs[-1],
                self.buffer[a_i].rnn_states_critic[-1],
                self.buffer[a_i].masks[-1])
            next_value = torch2numpy(next_value)
            self.buffer[a_i].compute_returns(
                next_value, self.trainer[a_i].value_normalizer)

    def train(self, warmup=False):
        # Compute last value
        self.compute_last_value()
        # Train
        self.prep_training()
        train_infos = []
        for a_i in range(self.n_agents):
            train_info = self.trainer[a_i].train(
                self.buffer[a_i], warmup=warmup)
            train_infos.append(train_info)
        return train_infos

    def get_save_dict(self):
        self.prep_rollout("cpu")
        agents_params = []
        for a_i in range(self.n_agents):
            params = {
                "actor": self.trainer[a_i].policy.actor.state_dict(),
                "critic": self.trainer[a_i].policy.critic.state_dict()
            }
            if self.trainer[a_i]._use_valuenorm:
                params["vnorm"] = self.trainer[a_i].value_normalizer.state_dict()
            agents_params.append(params)
        save_dict = {
            "agents_params": agents_params
        }
        return save_dict

    def load_params(self, agent_params):
        for a_i in range(self.n_agents):
            self.trainer[a_i].policy.actor.load_state_dict(
                agent_params[a_i]["actor"])
            self.trainer[a_i].policy.critic.load_state_dict(
                agent_params[a_i]["critic"])
            if self.trainer[a_i]._use_valuenorm:
                self.trainer[a_i].value_normalizer.load_state_dict(
                    agent_params[a_i]["vnorm"])

    def save(self, path):
        save_dict = self.get_save_dict()
        torch.save(save_dict, path)

