import torch
import numpy as np

from .qmix import QMIX
from .lnoveld import NovelD


class QMIX_MANovelD(QMIX):
    """ 
    Class impelementing MADDPG with Multi-Agent NovelD (MADDPG_MANovelD),
    meaning that we use a single NovelD model to compute the intrinsic reward
    of the multi-agent system.
    """
    def __init__(self, nb_agents, obs_dim, act_dim, lr, 
                 gamma=0.99, tau=0.01, hidden_dim=64, shared_params=False, 
                 init_explo_rate=1.0, max_grad_norm=None, device="cpu",
                 embed_dim=16, nd_lr=1e-4, nd_scale_fac=0.5):
        super(QMIX_MANovelD, self).__init__(
            nb_agents, obs_dim, act_dim, lr, gamma, tau, hidden_dim, 
            shared_params, init_explo_rate, max_grad_norm, device)
        # Init NovelD model for the multi-agent system
        self.ma_noveld = NovelD(
            nb_agents * obs_dim, embed_dim, hidden_dim, nd_lr, nd_scale_fac)

    def get_actions(self, 
            obs_list, last_actions, qnets_hidden_states, explore=False):
        # If we are starting a new episode, compute novelty for first obs
        if self.ma_noveld.is_empty():
            cat_obs = torch.Tensor(
                np.concatenate(obs_list)).unsqueeze(0).to(self.device)
            self.ma_noveld.get_reward(cat_obs.view(1, -1))
        return super().get_actions(
            obs_list, last_actions, qnets_hidden_states, explore)

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
        cat_obs = torch.Tensor(
            np.concatenate(next_obs_list)).unsqueeze(0).to(self.device)
        # Get reward
        int_reward = self.ma_noveld.get_reward(cat_obs)
        int_rewards = [int_reward] * self.nb_agents
        return int_rewards
    
    def train_on_batch(self, batch):
        """
        Update all agents and NovelD model.
        Inputs:
            batch (tuple(torch.Tensor)): Tuple of batches of experiences for
                the agents to train on.
        Outputs:
            qtot_loss (float): QMIX loss.
            nd_loss (float): MA-NovelD loss.
        """
        qtot_loss = super().train_on_batch(batch)

        # NovelD update
        nd_loss = self.ma_noveld.train_predictor()

        return qtot_loss, nd_loss

    def reset_noveld(self):
        self.ma_noveld.init_new_episode()
    
    def prep_training(self, device='cpu'):
        super().prep_training(device)
        self.ma_noveld.set_train(device)
    
    def prep_rollouts(self, device='cpu'):
        super().prep_rollouts(device)
        self.ma_noveld.set_eval(device)

    def save(self, filename):
        self.prep_training(device='cpu')
        save_dict = {
            'nb_agents': self.nb_agents,
            'obs_dim': self.obs_dim,
            'act_dim': self.act_dim,
            'lr': self.lr,
            'gamma': self.gamma,
            'tau': self.tau,
            'hidden_dim': self.hidden_dim,
            'shared_params': self.shared_params,
            'max_grad_norm': self.max_grad_norm,
            'agent_params': [a.get_params() for a in self.agents],
            'mixer_params': self.mixer.state_dict(),
            'target_mixer_params': self.target_mixer.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'manoveld_params': self.ma_noveld.get_params()
        }
        torch.save(save_dict, filename)