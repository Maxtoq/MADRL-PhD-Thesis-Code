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
    
    def init_new_episode(self):
        if self.rnd is not None:
            self.last_nov = None
        if self.e3b is not None:
            self.e3b.init_new_episode()
    
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
        
    def get_reward(self, state):
        """
        Get intrinsic reward for the given state.
        Inputs:
            state (torch.Tensor): State from which to generate the reward,
                dim=(1, state_dim).
        Outputs:
            int_reward (float): Intrinsic reward for the input state.
        """
        ## NovelD
        if self.rnd is not None:
            # Get RND reward as novelty
            nov = self.rnd.get_reward(state)

            # Compute reward
            if self.last_nov is not None:
                int_reward = max(nov - self.scale_fac * self.last_nov, 0.0)
            else:
                int_reward = 0.0

            self.last_nov = nov
        else:
            int_reward = 1.0

        ## E3B
        if self.e3b is not None:
            elliptic_scale = self.e3b.get_reward(state)
            elliptic_scale = math.sqrt(2 * elliptic_scale)
        else:
            elliptic_scale = 1.0

        return int_reward * elliptic_scale

    def train(self, state_batch, act_batch):
        """
        Inputs:
            state_batch (torch.Tensor): Batch of states, dim=(episode_length, 
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
        return rnd_loss + e3b_loss
    
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
