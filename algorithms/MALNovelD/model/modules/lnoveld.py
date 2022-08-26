import torch

from torch import nn
from torch.optim import Adam

from .networks import MLPNetwork
from .lm import GRUEncoder


class NovelD:
    """
    Class implementing NovelD, from "NovelD: A Simple yet Effective Exploration
    Criterion" (Zhang et al., 2021).
    """
    def __init__(self, state_dim, embed_dim, hidden_dim, 
                 lr=1e-4, scale_fac=0.5):
        """
        Inputs:
            :param state_dim (int): Dimension of the input.
            :param embed_dim (int): Dimension of the output of RND networks.
            :param hidden_dim (int): Dimension of the hidden layers in MLPs.
            :param lr (float): Learning rate for training the predictor
                (default=0.0001).
            :param scale_fac (float): Scaling factor for computing the reward, 
                noted alpha in the paper, controls how novel we want the states
                to be to generate some reward (in [0,1]) (default=0.5).
        """
        # Random Network Distillation network
        # Fixed target embedding network
        self.target = MLPNetwork(
            state_dim, embed_dim, hidden_dim, n_layers=3, norm_in=False)
        # Predictor embedding network
        self.predictor = MLPNetwork(
            state_dim, embed_dim, hidden_dim, n_layers=3, norm_in=False)

        # Fix weights of target
        for param in self.target.parameters():
            param.requires_grad = False

        # Training objects
        self.lr = lr
        self.predictor_loss = nn.MSELoss()
        self.predictor_optim = Adam(self.predictor.parameters(), lr=lr)
        
        # Scaling facor
        self.scale_fac = scale_fac
        
        # Last state novelty
        self.last_nov = None

        # Save count of states encountered during each episode
        self.episode_states_count = {}

        # Stored predictions for future training
        self.stored_preds = torch.Tensor()
        self.stored_targets = torch.Tensor()

    def init_new_episode(self):
        self.last_nov = None
        self.episode_states_count = {}

    def is_empty(self):
        return True if len(self.episode_states_count) == 0 else False
    
    def store_pred(self, pred, target):
        self.stored_preds = torch.cat((self.stored_preds, pred))
        self.stored_targets = torch.cat((self.stored_targets, target))

    def train_predictor(self):
        self.predictor_optim.zero_grad()
        loss = self.predictor_loss(self.stored_preds, self.stored_targets)
        loss.backward()
        self.predictor_optim.step()

        self.stored_preds = torch.Tensor()
        self.stored_targets = torch.Tensor()

        return float(loss)

    def get_reward(self, state):
        """
        Get intrinsic reward for this new state.
        Inputs:
            state (torch.Tensor): State from which to generate the reward,
                dim=(1, state_dim).
        Outputs:
            intrinsic_reward (float): Intrinsic reward for the input state.
        """
        # Increment count of current state
        state_key = tuple(state.squeeze().tolist())
        if state_key in self.episode_states_count:
            self.episode_states_count[state_key] += 1
        else:
            self.episode_states_count[state_key] = 1

        # Compute embeddings
        target = self.target(state)
        pred = self.predictor(state)

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

        # Store predictions
        self.store_pred(pred, target)

        return intrinsic_reward

    def get_params(self):
        return {'target': self.target.state_dict(),
                'predictor': self.predictor.state_dict(),
                'predictor_optim': self.predictor_optim.state_dict()}



class LNovelD:
    """
    Class implementing the Language-augmented version of NovelD from "Improving
    Intrinsic Exploration with Language Abstractions" (Mu et al., 2022).
    """
    def __init__(self, 
            obs_in_dim, 
            lang_in_dim,
            embed_dim,
            hidden_dim=64,
            lr=1e-4, 
            scale_fac=0.5, 
            trade_off=1):
        """
        Inputs:
            :param obs_in_dim (int): Dimension of the observation input.
            :param lang_in_dim (int): Dimension of the language encoding input.
            :param embed_dim (int): Dimension of the output of RND networks.
            :param lang_encoder (.lm.GRUEncoder): Network for encoding input
                sentences.
            :param hidden_dim (int): Dimension of the hidden layers in MLPs.
            :param lr (float): Learning rates for training NovelD predictors.
            :param scale_fac (float): Scaling factor for computing the reward, 
                noted alpha in the paper, controls how novel we want the states
                to be to generate some reward (in [0,1]).
            :param trade_off (float): Parameter for controlling the weight of 
                the language novelty in the final reward, noted lambda_l in the
                paper (in [0, +inf]).
        """
        # Observatio-based NovelD
        self.obs_noveld = NovelD(
            obs_in_dim, embed_dim, hidden_dim, lr, scale_fac)
        # Language-based NovelD
        self.lang_noveld = NovelD(
            lang_in_dim, embed_dim, hidden_dim, lr, scale_fac)

        self.trade_off = trade_off

    def is_empty(self):
        return self.obs_noveld.is_empty() or self.lang_noveld.is_empty()

    def reset(self):
        self.obs_noveld.init_new_episode()
        self.lang_noveld.init_new_episode()

    def get_reward(self, obs_in, lang_in):
        obs_int_reward = self.obs_noveld.get_reward(obs_in)
        lang_int_reward = self.lang_noveld.get_reward(lang_in)

        intrinsic_reward = obs_int_reward + self.trade_off * lang_int_reward

        return intrinsic_reward

    def train(self):
        obs_loss = self.obs_noveld.train_predictor()
        lang_loss = self.lang_noveld.train_predictor()
        return obs_loss, lang_loss
