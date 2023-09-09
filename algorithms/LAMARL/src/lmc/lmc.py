import copy
import torch
import numpy as np

from torch import nn

from .modules.lang_learner import LanguageLearner
from .modules.comm_policy import PerfectComm
from .policy.mappo.mappo import MAPPO


class LMC:
    """
    Language-Memory for Communication using a pre-defined discrete language.
    """
    def __init__(self, args, n_agents, obs_space, shared_obs_space, act_space, 
                 vocab, device):
        self.args = args
        self.n_agents = n_agents
        self.context_dim = args.context_dim
        self.n_parallel_envs = args.n_parallel_envs
        self.device = device

        # Modules
        self.lang_learner = LanguageLearner(
            obs_space[0].shape[0], 
            self.context_dim, 
            args.lang_hidden_dim, 
            vocab, 
            device,
            args.lang_lr,
            args.lang_n_epochs,
            args.lang_batch_size)

        if args.comm_policy_algo == "perfect_comm":
            self.comm_policy = PerfectComm(self.lang_learner)
            # CommunicationPolicy(context_dim, hidden_dim)
        elif args.comm_policy_algo == "no_comm":
            self.comm_policy = None
            self.context_dim = 0
        else:
            raise NotImplementedError("Bad name given for communication policy algo.")

        if args.policy_algo == "mappo":
            self.policy = MAPPO(
                args, n_agents, obs_space, shared_obs_space, self.context_dim,
                act_space, device)

        self.message_context = np.zeros(
            (self.n_parallel_envs, self.context_dim))

    def prep_training(self):
        self.lang_learner.prep_training()
        self.policy.prep_training()

    def prep_rollout(self, device=None):
        self.lang_learner.prep_rollout(device)
        self.policy.prep_rollout(device)

    def start_episode(self, obs):
        self.policy.start_episode(obs, self.message_context)

    def comm_n_act(self, obs, perfect_messages=None):
        # Get actions
        pol_outputs = self.policy.get_actions()
        # Get messages
        if self.comm_policy is not None:
            broadcasts, next_contexts = self.comm_policy.comm_step(
                obs, perfect_messages)
            self.message_context = next_contexts
        else:
            next_contexts, broadcasts = np.zeros(
            (self.n_parallel_envs, self.context_dim)), []

        return pol_outputs + (next_contexts, broadcasts)

    def reset_context(self, env_dones):
        """
        :param env_dones (list(bool)): Done state for each parallel environment.
        """
        self.message_context = \
            self.message_context * (1 - env_dones)[..., np.newaxis]

    def store_exp(self, data):
        self.policy.store(data)

    def train(self):
        # Train policy
        pol_losses = self.policy.train()
        # Train language
        
        return pol_losses

    def save(self):
        pass
