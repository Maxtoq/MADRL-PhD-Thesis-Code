import torch
import numpy as np

from .language.lang_learner import LanguageLearner
from .policy.acc_mappo import ACC_MAPPO


class LanguageGroundedMARL:

    def __init__(self, args, n_agents, obs_space, shared_obs_space, act_space, 
                 vocab, device="cpu", comm_logger=None):
        self.args = args
        self.context_dim = args.context_dim
        self.n_parallel_envs = args.n_parallel_envs
        self.n_warmup_steps = args.n_warmup_steps
        self.comm_n_warmup_steps = args.comm_n_warmup_steps
        self.token_penalty = args.comm_token_penalty
        self.env_reward_coef = args.comm_env_reward_coef
        self.comm_logger = comm_logger
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

        self.comm_n_act_policy = ACC_MAPPO(
            args, 
            lang_learner, 
            n_agents, 
            obs_space, 
            shared_obs_space, 
            act_space[0], 
            device)

    def prep_training(self):
        self.lang_learner.prep_training()
        self.comm_n_act_policy.prep_training()

    def prep_rollout(self, device=None):
        self.lang_learner.prep_rollout(device)
        self.comm_n_act_policy.prep_rollout(device)

        