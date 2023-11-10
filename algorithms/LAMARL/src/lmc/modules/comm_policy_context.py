import copy
import torch
import random
import itertools
import numpy as np

from torch import nn
from gym import spaces

from src.lmc.modules.networks import MLPNetwork, init
from src.lmc.policy.mappo.mappo_shared import MAPPO
from src.lmc.utils import get_mappo_args


def torch2numpy(x):
    return x.detach().cpu().numpy()


class CommPol_Context:
    """ 
    Communication module with a recurrent context encoder, 
    a policy that generates sentences and a value that estimates
    the quality of the current state (previous hidden state).
    It is trained using PPO, fine-tuning a pretrained policy.
    """
    def __init__(self, args, n_agents, lang_learner, device="cpu"):
        self.n_agents = n_agents
        self.n_envs = args.n_parallel_envs
        self.obs_dist_coef = args.comm_obs_dist_coef
        self.device = device
        self.warming_up = False
        
        self.lang_learner = lang_learner

        comm_policy_args = get_mappo_args(args)
        context_dim = args.context_dim
        input_dim = context_dim * 2
        low = np.full(context_dim, -np.inf)
        high = np.full(context_dim, np.inf)
        act_space = spaces.Box(low, high)
        self.context_encoder_policy = MAPPO(
            comm_policy_args, 
            n_agents, 
            input_dim, 
            input_dim * n_agents, 
            act_space, 
            device)

        self.values = None
        self.comm_context = None
        self.action_log_probs = None
        self.rnn_states = None
        self.critic_rnn_states = None
        self.obs_dist = None

    def prep_rollout(self, device=None):
        if device is None:
            device = self.device
        self.context_encoder_policy.prep_rollout(device)

    def prep_training(self):
        self.context_encoder_policy.prep_training()

    def start_episode(self):
        self.context_encoder_policy.start_episode()
        
    @torch.no_grad()
    def get_messages(self, obs, lang_context):
        """
        Perform a communication step: encodes obs and previous messages and
        generates messages for this step.
        :param obs: (np.ndarray) agents' observations for all parallel 
            environments, dim=(n_envs, n_agents, obs_dim)
        :param lang_context: (np.ndarray) Language contexts from last step, 
            dim=(n_envs, n_agents, context_dim)
            
        :return messages (list(list(str))): messages generated for each agent,
            for each parallel environment
        """
        # Encode inputs
        # obs_context = []
        obs = torch.Tensor(obs).view(self.n_envs * self.n_agents, -1)
        obs_context = self.lang_learner.encode_observations(obs)
        obs_context = obs_context.view(self.n_envs, self.n_agents, -1)

        # Repeat lang_contexts for each agent in envs
        lang_context = torch.from_numpy(lang_context.repeat(
            self.n_agents, 0).reshape(
                self.n_envs, self.n_agents, -1)).to(self.device)

        input_context = torch.cat((obs_context, lang_context), dim=-1)
        
        # Make all possible shared inputs
        shared_input = []
        ids = list(range(self.n_agents)) * 2
        for a_i in range(self.n_agents):
            shared_input.append(
                input_context[:, ids[a_i:a_i + self.n_agents]].reshape(
                    self.n_envs, 1, -1))
        shared_input = torch.cat(shared_input, dim=1)

        self.context_encoder_policy.store_obs(
            torch2numpy(input_context), torch2numpy(shared_input))

        self.values, self.comm_context, self.action_log_probs, self.rnn_states, \
            self.critic_rnn_states = self.context_encoder_policy.get_actions()

        # Compute distance from observation context
        self.obs_dist = np.linalg.norm(
            torch2numpy(obs_context) - self.comm_context, 2, axis=-1)

        messages = self.lang_learner.generate_sentences(torch.Tensor(
            self.comm_context).view(self.n_envs * self.n_agents, -1).to(
                self.device))
        
        return messages

    # def _rand_filter_messages(self, messages):
    #     """
    #     Randomly filter out perfect messages.
    #     :param messages (list(list(list(str)))): Perfect messages, ordered by
    #         environment, by agent.

    #     :return filtered_broadcast (list(list(str))): Filtered message to 
    #         broadcast, one for each environment.
    #     """
    #     filtered_broadcast = []
    #     for env_messages in messages:
    #         env_broadcast = []
    #         for message in env_messages:
    #             if random.random() < 0.2:
    #                 env_broadcast.extend(message)
    #         filtered_broadcast.append(env_broadcast)
    #     return filtered_broadcast
    
    @torch.no_grad()
    def comm_step(self, obs, lang_contexts, perfect_messages=None):
        # Get messages
        messages = self.get_messages(obs, lang_contexts)
        
        # Arrange messages by env
        broadcasts = []
        messages_by_env = []
        for e_i in range(self.n_envs):
            env_broadcast = []
            for a_i in range(self.n_agents):
                env_broadcast.extend(messages[e_i * self.n_agents + a_i])
            broadcasts.append(env_broadcast)
            messages_by_env.append(messages[
                e_i * self.n_agents:e_i * self.n_agents + self.n_agents])

        new_lang_contexts = self.lang_learner.encode_sentences(
            broadcasts).cpu().numpy()

        # # TEST with perfect messages
        # broadcasts = self._rand_filter_messages(perfect_messages)
        # new_lang_contexts = self.lang_learner.encode_sentences(broadcasts).detach().cpu().numpy()
        
        # Return messages and lang_context
        return broadcasts, messages_by_env, new_lang_contexts
    
    def store_rewards(self, message_rewards, dones):
        """
        Send rewards for each sentences to the buffer to compute returns.
        :param message_rewards (np.ndarray): Rewards for each generated 
            sentence, dim=(n_envs, n_agents, 1).
        :param dones (np.ndarray): Done state of each environment, 
            dim=(n_envs, n_agents).

        :return rewards (dict): Rewards to log.
        """
        message_rewards -= self.obs_dist[..., np.newaxis] \
                            * self.obs_dist_coef

        self.context_encoder_policy.store_act(
            message_rewards, dones, 
            self.values, 
            self.comm_context, 
            self.action_log_probs, 
            self.rnn_states, 
            self.critic_rnn_states)

        rewards = {
            "message_reward": message_rewards.mean(),
            "obs_distance": float(self.obs_dist.mean())
        }

        return rewards
    
    def train(self, warmup=False):
        losses = self.context_encoder_policy.train(warmup)
        return losses

    def get_save_dict(self):
        save_dict = {
            "context_encoder": self.context_encoder_policy.get_save_dict()}
        return save_dict

    def load_params(self, save_dict):
        self.lang_learner.load_params(save_dict)
        if "context_encoder" in save_dict:
            self.context_encoder_policy.load_params(
                save_dict["context_encoder"]["agents_params"])
