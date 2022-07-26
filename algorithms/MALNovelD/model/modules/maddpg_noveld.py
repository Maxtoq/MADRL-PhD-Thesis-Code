import torch

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

    def step(self, obs, explore):
        # If we are starting a new episode, compute novelty for first observation
        if self.noveld.is_empty():
            self.noveld.get_reward(obs)

        return super().step(obs, explore)

    def get_intrinsic_reward(self, next_obs):
        intr_reward, _ = self.noveld.get_reward(next_obs)
        return intr_reward


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

