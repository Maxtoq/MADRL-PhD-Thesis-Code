import math
import torch
from torch.nn import functional as F

from .networks import MLPNetwork
from .intrinsic_rewards import IntrinsicReward
from .rnd import RND
from .e3b import E3B


class E2S_NovelD(IntrinsicReward):
    """ Elliptical Episodic Scaling of NovelD. """

    def __init__(self, input_dim, act_dim, enc_dim, hidden_dim, 
                 scale_fac=0.5, ridge=0.1, lr=1e-4, device="cpu", 
                 ablation=None):
        assert ablation in [None, "LLEC", "EEC"], "Wrong ablation name, must be in [None, 'LLEC', 'EEC']"
        self.ablation = ablation
        # NovelD parameters
        self.scale_fac = scale_fac
        self.last_nov = None
        # Models
        self.rnd = None
        self.e3b = None
        if self.ablation in [None, "LLEC"]:
            self.rnd = RND(input_dim, enc_dim, hidden_dim, lr, device)
        if self.ablation in [None, "EEC"]:
            self.e3b = E3B(
                input_dim, act_dim, enc_dim, hidden_dim, ridge, lr, device)
    
    def init_new_episode(self, n_episodes=1):
        if self.rnd is not None:
            self.last_nov = None
        if self.e3b is not None:
            self.e3b.init_new_episode(n_episodes)
    
    def set_train(self, device):
        if self.rnd is not None:
            self.rnd.set_train(device)
        if self.e3b is not None:
            self.e3b.set_train(device)

    def set_eval(self, device):
        if self.rnd is not None:
            self.rnd.set_eval(device)
        if self.e3b is not None:
            self.e3b.set_eval(device)
        
    def get_reward(self, state_batch):
        """
        Get intrinsic reward for the given state.
        Inputs:
            state_batch (torch.Tensor): States from which to generate the 
                rewards, dim=(batch_size, state_dim).
        Outputs:
            ir_reward (torch.Tensor): Intrinsic rewards for the input states,
                dim=(batch_size, 1).
        """
        batch_size = state_batch.shape[0]
        ## NovelD
        if self.rnd is not None:
            # Get RND reward as novelty
            nov = self.rnd.get_reward(state_batch)

            # Compute reward
            if self.last_nov is not None:
                noveld_reward = torch.max(
                    nov - self.scale_fac * self.last_nov, 
                    torch.zeros(batch_size))
            else:
                noveld_reward = torch.Tensor([0.0] * batch_size)

            self.last_nov = nov
        else:
            noveld_reward = torch.Tensor([1.0] * batch_size)

        ## E3B
        if self.e3b is not None:
            elliptic_scale = self.e3b.get_reward(state_batch)
            elliptic_scale = torch.sqrt(2 * elliptic_scale)
        else:
            elliptic_scale = torch.Tensor([1.0] * batch_size)

        return noveld_reward * elliptic_scale

    def train(self, state_batch, act_batch):
        """
        Inputs:
            state_batch (torch.Tensor): Batch of states, dim=(episode_length + 1,
                batch_size, state_dim).
            act_batch (torch.Tensor): Batch of actions, dim=(episode_length, 
                batch_size, action_dim).
        """
        rnd_loss = 0.0
        e3b_loss = 0.0
        if self.rnd is not None:
            rnd_loss = self.rnd.train(state_batch, act_batch)
        if self.e3b is not None:
            e3b_loss = self.e3b.train(state_batch, act_batch)
        return {"rnd_loss": rnd_loss, "e3b_loss": e3b_loss}
    
    def get_params(self):
        rnd_params = {}
        e3b_params = {}
        if self.rnd is not None:
            rnd_params = self.rnd.get_params()
        if self.e3b is not None:
            e3b_params = self.e3b.get_params()
        return dict(rnd_params, **e3b_params)

    def load_params(self, params):
        if self.rnd is not None:
            self.rnd.load_params(params)
        if self.e3b is not None:
            self.e3b.load_params(params)
