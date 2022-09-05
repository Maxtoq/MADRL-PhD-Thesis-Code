import torch
import numpy as np

from .maddpg import DDPGAgent, MADDPG
from .lnoveld import NovelD


class DDPG_NovelD(DDPGAgent):

    def __init__(self, policy_in_dim, policy_out_dim, critic_in_dim, lr, 
                 embed_dim, hidden_dim=64, discrete_action=False, 
                 init_explo=1.0, explo_strat='sample', nd_lr=1e-4, 
                 nd_scale_fac=0.5):
        super(DDPG_NovelD, self).__init__(
            policy_in_dim, policy_out_dim, critic_in_dim, lr, hidden_dim, 
            discrete_action, init_explo, explo_strat)
        self.noveld = NovelD(
            policy_in_dim, embed_dim, hidden_dim, nd_lr, nd_scale_fac)

    def step(self, obs, explore, device="cpu"):
        # If we are starting a new episode, compute novelty for first observation
        if self.noveld.is_empty():
            self.noveld.get_reward(obs)

        return super().step(obs, explore, device)

    def get_intrinsic_reward(self, next_obs):
        intr_reward = self.noveld.get_reward(next_obs)
        return intr_reward
    
    def train_noveld(self):
        return self.noveld.train_predictor()

    def reset_noveld(self):
        self.noveld.init_new_episode()


class MADDPG_PANovelD(MADDPG):
    """ 
    Class impelementing MADDPG with Per Agent NovelD (MADDPG_PANovelD),
    meaning that each agent has its own local NovelD model to compute a
    personal intrinsic reward.
    """
    def __init__(self, n_agents, input_dim, act_dim, lr=0.0007, gamma=0.95, 
                 tau=0.01, hidden_dim=64, embed_dim=16, discrete_action=False, 
                 shared_params=False, init_explo_rate=1.0, explo_strat="sample",
                 nd_lr=1e-4, nd_scale_fac=0.5):
        super(MADDPG_PANovelD, self).__init__(
            n_agents, input_dim, act_dim, lr, gamma, tau, hidden_dim, 
            discrete_action, shared_params, init_explo_rate, explo_strat)
        # Create agent models
        critic_input_dim = n_agents * input_dim + n_agents * act_dim
        if not shared_params:
            self.agents = [DDPG_NovelD(
                    input_dim, act_dim, critic_input_dim, lr, embed_dim, 
                    hidden_dim, discrete_action, init_explo_rate, explo_strat, 
                    nd_lr, nd_scale_fac)
                for _ in range(n_agents)]
        else:
            self.agents = [DDPG_NovelD(
                    input_dim, act_dim, critic_input_dim, lr, embed_dim, 
                    hidden_dim, discrete_action, init_explo_rate, explo_strat, 
                    nd_lr, nd_scale_fac)]
        
    def get_intrinsic_rewards(self, next_obs_list):
        """
        Get intrinsic rewards for all agents.
        Inputs:
            next_obs_list (list): List of agents' observations at next 
                step.
        Outputs:
            int_rewards (list): List of agents' intrinsic rewards.
        """
        int_rewards = []
        for a_i, next_obs in enumerate(next_obs_list):
            a_i = 0 if self.shared_params else a_i
            int_reward = self.agents[a_i].get_intrinsic_reward(
                torch.Tensor(next_obs).unsqueeze(0))
            int_rewards.append(int_reward)
        return int_rewards

    def update(self, samples):
        vf_losses = []
        pol_losses = []
        nd_losses = []
        for a_i, sample in enumerate(samples):
            a_i = 0 if self.shared_params else a_i
            # Agent update
            vf_loss, pol_loss = super().update(sample, a_i)
            # NovelD update
            nd_loss = self.agents[a_i].train_noveld()
            vf_losses.append(vf_loss)
            pol_losses.append(pol_loss)
            nd_losses.append(nd_loss)

        return vf_losses, pol_losses, nd_losses

    def reset_noveld(self):
        for a_i in range(self.n_agents):
            a_i = 0 if self.shared_params else a_i
            self.agents[a_i].reset_noveld()


class MADDPG_MANovelD(MADDPG):
    """ 
    Class impelementing MADDPG with Multi-Agent NovelD (MADDPG_MANovelD),
    meaning that we use a single NovelD model to compute the intrinsic reward
    of the multi-agent system.
    """
    def __init__(self, n_agents, input_dim, act_dim, lr=0.0007, gamma=0.95, 
                 tau=0.01, hidden_dim=64, embed_dim=16, discrete_action=False, 
                 shared_params=False, init_explo_rate=1.0, explo_strat="sample",
                 nd_lr=1e-4, nd_scale_fac=0.5):
        super(MADDPG_MANovelD, self).__init__(
            n_agents, input_dim, act_dim, lr, gamma, tau, hidden_dim, 
            discrete_action, shared_params, init_explo_rate, explo_strat)

        # Init NovelD model for the multi-agent system
        self.ma_noveld = NovelD(
            n_agents * input_dim, embed_dim, hidden_dim, nd_lr, nd_scale_fac)

    def step(self, observations, explore=False):
        # If we are starting a new episode, compute novelty for first observation
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


class MADDPG_MPANovelD(MADDPG):
    """ 
    Class impelementing MADDPG with Multi & Per-Agent NovelD (MADDPG_MPANovelD),
    meaning that there is one NovelD model for the whole multi-agent system and 
    one NovelD model for each agent.
    """
    def __init__(self, n_agents, input_dim, act_dim, lr=0.0007, gamma=0.95, 
                 tau=0.01, hidden_dim=64, embed_dim=16, discrete_action=False, 
                 shared_params=False, init_explo_rate=1.0, explo_strat="sample",
                 nd_lr=1e-4, nd_scale_fac=0.5):
        super(MADDPG_MPANovelD, self).__init__(
            n_agents, input_dim, act_dim, lr, gamma, tau, hidden_dim, 
            discrete_action, shared_params, init_explo_rate, explo_strat)

        # Init NovelD model for the multi-agent system
        self.ma_noveld = NovelD(
            n_agents * input_dim, embed_dim, hidden_dim, nd_lr, nd_scale_fac)

        # Init agents with their NovelD model
        critic_input_dim = n_agents * input_dim + n_agents * act_dim
        if not shared_params:
            self.agents = [DDPG_NovelD(
                    input_dim, act_dim, critic_input_dim, lr, embed_dim, 
                    hidden_dim, discrete_action, init_explo_rate, explo_strat, 
                    nd_lr, nd_scale_fac)
                for _ in range(n_agents)]
        else:
            self.agents = [DDPG_NovelD(
                    input_dim, act_dim, critic_input_dim, lr, embed_dim, 
                    hidden_dim, discrete_action, init_explo_rate, explo_strat, 
                    nd_lr, nd_scale_fac)]

    def step(self, observations, explore=False):
        # If we are starting a new episode, compute novelty for first observation
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
        # Local intrinsic rewards for each agent
        int_rewards = np.zeros(2)
        for a_i, next_obs in enumerate(next_obs_list):
            a_i = 0 if self.shared_params else a_i
            local_int_reward = self.agents[a_i].get_intrinsic_reward(
                torch.Tensor(next_obs).unsqueeze(0))
            int_rewards[a_i] = local_int_reward

        # Global instrinsic rewards
        # Concatenate observations
        cat_obs = torch.Tensor(np.concatenate(next_obs_list)).unsqueeze(0)
        # Get reward
        global_int_reward = self.ma_noveld.get_reward(cat_obs)
        int_rewards += global_int_reward
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
        nd_losses = []
        for a_i, sample in enumerate(samples):
            a_i = 0 if self.shared_params else a_i
            # Agent update
            vf_loss, pol_loss = super().update(sample, a_i)
            # NovelD update
            nd_loss = self.agents[a_i].train_noveld()
            vf_losses.append(vf_loss)
            pol_losses.append(pol_loss)
            nd_losses.append(nd_loss)

        # NovelD update
        mand_loss = self.ma_noveld.train_predictor()
        mand_losses = [mand_loss] * self.n_agents

        return vf_losses, pol_losses, nd_losses, mand_losses

    def reset_noveld(self):
        for a_i in range(self.n_agents):
            a_i = 0 if self.shared_params else a_i
            self.agents[a_i].reset_noveld()
            
        self.ma_noveld.init_new_episode()