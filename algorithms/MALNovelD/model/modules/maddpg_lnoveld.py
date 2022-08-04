import torch
import numpy as np

from .maddpg import DDPGAgent, MADDPG
from .lnoveld import LNovelD


class MADDPG_MALNovelD(MADDPG):
    """ 
    Class impelementing MADDPG with Multi-Agent Language-NovelD 
    (MADDPG_MANovelD), meaning that we use a single L-NovelD model to compute 
    the intrinsic reward of the multi-agent system.
    """
    def __init__(self, n_agents, obs_dim, lang_dim, act_dim, lr=0.0007, 
                 gamma=0.95, tau=0.01, hidden_dim=64, embed_dim=16, 
                 discrete_action=False, shared_params=False, 
                 init_explo_rate=1.0, explo_strat="sample",
                 nd_lr=1e-4, nd_scale_fac=0.5, lang_trade_off=1):
        super(MADDPG_MALNovelD, self).__init__(
            n_agents, obs_dim, act_dim, lr, gamma, tau, hidden_dim, 
            discrete_action, shared_params, init_explo_rate, explo_strat)

        # Init NovelD model for the multi-agent system
        self.ma_lnoveld = LNovelD(
            n_agents * obs_dim, 
            n_agents * lang_dim, 
            embed_dim, 
            hidden_dim, 
            nd_lr, 
            nd_scale_fac,
            lang_trade_off)

    def step(self, observations, sentences, explore=False):
        # If we are starting a new episode, compute novelty for first 
        # observation
        if self.ma_noveld.is_empty():
            self.ma_noveld.get_reward(observations.view(1, -1))

        return super().step(observations, explore)

    def get_intrinsic_rewards(self, next_obs_list):
        """
        Get intrinsic reward of the multi-agent system.
        Inputs:
            next_obs_list (list): List of agents' observations at next 
                step.
        Outputs:
            int_rewards (list): List of agents' intrinsic rewards.
        """
        # Concatenate observations
        cat_obs = torch.Tensor(np.concatenate(next_obs_list)).unsqueeze(0)
        # Get reward
        int_reward = self.ma_noveld.get_reward(cat_obs)
        int_rewards = [int_reward] * self.n_agents
        return int_rewards

    def update(self, samples):
        """
        Update all agents and NovelD model.
        Inputs:
            samples (list): List of batches for the agents to train on (one 
                for each agent).
            logger (tensorboardX.SummaryWriter): Tensorboad logger.
        """
        vf_losses = []
        pol_losses = []
        for a_i, sample in enumerate(samples):
            a_i = 0 if self.shared_params else a_i
            vf_loss, pol_loss = super().update(sample, a_i)
            vf_losses.append(vf_loss)
            pol_losses.append(pol_loss)

        # NovelD update
        nd_losses = [self.ma_noveld.train_predictor()] * self.n_agents

        return vf_losses, pol_losses, nd_losses

    def reset_noveld(self):
        self.ma_noveld.init_new_episode()

