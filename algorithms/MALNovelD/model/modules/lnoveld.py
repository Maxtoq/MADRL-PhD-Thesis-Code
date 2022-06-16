import torch

from torch import nn

from .networks import MLPNetwork


class NovelD(nn.Module):
    """
    Class implementing NovelD, from "NovelD: A Simple yet Effective Exploration
    Criterion" (Zhang et al., 2021)
    """
    def __init__(self, state_dim, embed_dim, hidden_dim, scale_fac):
        """
        Inputs:
            :param state_dim (int): Dimension of the input
            :param embed_dim (int): Dimension of the output of RND networks
            :param hidden_dim (int): Dimension of the hidden layers in MLPs
            :param scale_fac (float): Scaling factor for computing the reward, 
                noted alpha in the paper, controls how novel we want the states
                to be to generate some reward (in [0,1])
        """
        super(NovelD, self).__init__()
        # Random Network Distillation network
        # Fixed target embedding network
        self.target = MLPNetwork(
            state_dim, embed_dim, hidden_dim, n_layers=3)
        # Predictor embedding network
        self.predictor = MLPNetwork(
            state_dim, embed_dim, hidden_dim, n_layers=3)

        # Fix weights of target
        for param in self.target.parameters():
            param.requires_grad = False
        
        # Scaling facor
        self.scale_fac = scale_fac
        
        # Last state novelty
        self.last_nov = None

        # Save count of states encountered during each episode
        self.episode_states_count = {}

    def forward(self, X):
        # Increment count of current state
        state_key = tuple(X.squeeze().tolist())
        if state_key in self.episode_states_count:
            self.episode_states_count[state_key] += 1
        else:
            self.episode_states_count[state_key] = 1

        # Compute embeddings
        target = self.target(X)
        pred = self.predictor(X)

        # Compute novelty
        nov = torch.norm(pred.detach() - target.detach(), dim=1, p=2)

        # Compute reward
        if self.episode_states_count[state_key] == 1 and \
            self.last_nov is not None:
            intrinsic_reward = float(torch.clamp(
                nov - self.scale_fac * self.last_nov, min=0))
        else:
            intrinsic_reward = 0.0

        self.last_nov = nov

        return target, pred, intrinsic_reward



class LNovelD(nn.Module):
    """
    Class implementing the Language-augmented version of NovelD from "Improving
    Intrinsic Exploration with Language Abstractions" (Mu et al., 2022).
    """
    def __init__(self, 
            obs_in_dim, 
            lang_in_dim,
            embed_dim,
            hidden_dim=64, 
            scale_fac=0.5, 
            trade_off=1):
        """
        Inputs:
            :param obs_in_dim (int): Dimension of the observation input
            :param lang_in_dim (int): Dimension of the language encoding input
            :param embed_dim (int): Dimension of the output of RND networks
            :param hidden_dim (int): Dimension of the hidden layers in MLPs
            :param scale_fac (float): Scaling factor for computing the reward, 
                noted alpha in the paper, controls how novel we want the states
                to be to generate some reward (in [0,1])
            :param trade_off (float): Parameter for controlling the weight of 
                the language novelty in the final reward, noted lambda_l in the
                paper (in [0, +inf])
        """
        super(LNovelD, self).__init__()
        # Observatio-based NovelD
        self.obs_noveld = NovelD(obs_in_dim, embed_dim, hidden_dim, scale_fac)
        # Language-based NovelD
        self.lang_noveld = NovelD(lang_in_dim, embed_dim, hidden_dim, scale_fac)

        self.trade_off = trade_off

    def forward(self, obs_in, lang_in):
        obs_target, obs_pred, obs_int_reward = self.obs_noveld(obs_in)
        lang_target, lang_pred, lang_int_reward = self.lang_noveld(lang_in)

        intrinsic_reward = obs_int_reward + self.trade_off * lang_int_reward

        return (obs_target, obs_pred), (lang_target, lang_pred), intrinsic_reward
