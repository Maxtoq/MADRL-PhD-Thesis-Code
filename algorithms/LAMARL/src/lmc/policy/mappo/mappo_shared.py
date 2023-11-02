import torch
import numpy as np

from .buffer import SharedReplayBuffer
from .policy import R_MAPPOPolicy
from .train_algo import R_MAPPOTrainAlgo

def torch2numpy(x):
    return x.detach().cpu().numpy()


class MAPPO:
    """
    Class handling training of MAPPO from paper "The Surprising Effectiveness 
    of PPO in Cooperative Multi-Agent Games" (https://arxiv.org/abs/2103.01955),
    with shared parameters accross all agents.
    :param args: (argparse.Namespace) all arguments for training
    :param n_agents: (int) number of agents
    :param obs_dim: (int) observation dimensions, for each agent
    :param cent_obs_dim: (int) centralized observation dimensions, for 
        each agent
    :param act_space: (gym.Space) action dimensions, for each agent
    :param device: (torch.device) cuda device used for training
    """
    def __init__(self, 
            args, n_agents, obs_dim, cent_obs_dim, act_space, device):
        self.args = args
        self.n_agents = n_agents
        self.n_parallel_envs = args["n_parallel_envs"]
        self.recurrent_N = args["recurrent_N"]
        self.use_centralized_V = self.args.use_centralized_V
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

        # Policy
        if self.use_centralized_V:
            shared_obs_dim = cent_obs_dim
        else:
            shared_obs_dim = obs_dim
        self.policy = R_MAPPOPolicy(
            self.args, obs_dim, shared_obs_dim, act_space, device)

        # Train algorithm
        self.trainer = R_MAPPOTrainAlgo(self.args, self.policy, device)

        # Replay buffer
        self.buffer = SharedReplayBuffer(
            self.args, n_agents, obs_dim, shared_obs_dim, act_space)

    def prep_rollout(self, device=None):
        if device is None:
            device = self.train_device
        self.trainer.prep_rollout(device)

    def prep_training(self):
        self.trainer.prep_training(self.train_device)

    def start_episode(self):
        self.buffer.reset_episode()

    @torch.no_grad()
    def get_actions(self):
        shared_obs, obs, rnn_states, critic_rnn_states, masks \
            = self.buffer.get_act_params()

        values, actions, action_log_probs, rnn_states, critic_rnn_states \
            = self.trainer.policy.get_actions(
                np.concatenate(shared_obs),
                np.concatenate(obs), 
                np.concatenate(rnn_states), 
                np.concatenate(critic_rnn_states), 
                np.concatenate(masks))

        values = np.reshape(
            torch2numpy(values), (self.n_parallel_envs, n_agents, -1))
        actions = np.reshape(
            torch2numpy(actions), (self.n_parallel_envs, n_agents, -1))
        action_log_probs = np.reshape(
            torch2numpy(action_log_probs), (self.n_parallel_envs, n_agents, -1))
        rnn_states = np.reshape(
            torch2numpy(rnn_states), 
            (self.n_parallel_envs, n_agents, self.recurrent_N, -1))
        critic_rnn_states = np.reshape(
            torch2numpy(critic_rnn_states), 
            (self.n_parallel_envs, n_agents, self.recurrent_N, -1))

        return values, actions, action_log_probs, rnn_states, \
               critic_rnn_states

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

