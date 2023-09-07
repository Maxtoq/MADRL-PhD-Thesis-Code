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
        else:
            raise NotImplementedError("Bad name given for communication policy algo.")

        if args.policy_algo == "mappo":
            self.policy = MAPPO(
                args, n_agents, obs_space, shared_obs_space, args.context_dim,
                act_space, device)

        self.device = device

    def prep_training(self):
        self.lang_learner.prep_training()
        self.policy.prep_training()

    def prep_rollout(self, device=None):
        self.lang_learner.prep_rollout(device)
        self.policy.prep_rollout(device)

    def start_episode(self, obs):
        # TODO: Ne pas mettre zeros pour le context sauf toute première step
        context = np.zeros((obs.shape[0], self.n_agents, self.context_dim))
        self.policy.start_episode(obs, context)

    def comm_n_act(self, obs, perfect_messages=None):
        # Get actions
        pol_outputs = self.policy.get_actions()
        # Get messages
        broadcasts, next_contexts = self.comm_policy.comm_step(
            obs, perfect_messages)

        return pol_outputs + (next_contexts, broadcasts)

    def store_exp(self, data):
        self.policy.store(data)

    def train(self):
        pass

    def save(self):
        pass
