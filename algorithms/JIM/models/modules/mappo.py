import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.distributions import Categorical

from .networks import MLPNetwork, get_init_linear
from .utils import soft_update


class PPOAgent:
    
    def __init__(self, obs_dim, act_dim, 
                 hidden_dim=64, init_explo=1.0, device="cpu"):
        self.epsilon = init_explo
        self.act_dim = act_dim
        self.hidden_dim = hidden_dim
        self.device = device

class MAPPO:

    def __init__(self, nb_agents, obs_dim, act_dim, lr, 
                 gamma=0.99, tau=0.01, hidden_dim=64, shared_params=False, 
                 init_explo_rate=1.0, max_grad_norm=None, device="cpu",
                 use_per=False, per_nu=0.9, per_eps=1e-6):
        self.nb_agents = nb_agents
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.lr = lr
        self.gamma = gamma
        self.tau = tau
        self.hidden_dim = hidden_dim
        self.shared_params = shared_params
        self.max_grad_norm = max_grad_norm
        self.device = device
        self.use_per = use_per
        self.per_nu = per_nu
        self.per_eps = per_eps

        # Create agent policies
        if not shared_params:
            self.agents = [PPOAgent(
                    obs_dim, 
                    act_dim,
                    hidden_dim, 
                    init_explo_rate,
                    device)
                for _ in range(nb_agents)]
        else:
            self.agents = [PPOAgent(
                    obs_dim, 
                    act_dim,
                    hidden_dim, 
                    init_explo_rate,
                    device)]
        