import torch
import torch.nn as nn

from .networks import MLPNetwork, get_init_linear


class QMixer(nn.Module):

    def __init__(self, n_agents, cent_obs_dim, device,
            mixer_hidden_dim=32, hypernet_hidden_dim=64):
        super(QMixer, self).__init__()
        self.n_agents = n_agents
        self.cent_obs_dim = cent_obs_dim
        self.device = device
        self.mixer_hidden_dim = mixer_hidden_dim
        self.hypernet_hidden_dim = hypernet_hidden_dim

        # Hypernets
        self.hypernet_weights1 = MLPNetwork(
            cent_obs_dim, n_agents * mixer_hidden_dim, hypernet_hidden_dim, 0)
        self.hypernet_bias1 = get_init_linear(cent_obs_dim, mixer_hidden_dim)
        self.hypernet_weights2 = MLPNetwork(
            cent_obs_dim, mixer_hidden_dim, hypernet_hidden_dim, 0)
        self.hypernet_bias2 = MLPNetwork(
            cent_obs_dim, 1, hypernet_hidden_dim, 0)

    def forward(self, local_qs, obs):
        pass

class QMIXAgent:

    def __init__(self, pol_in_dim, pol_out_dim, lr, 
                 hidden_dim=64, discrete_action=False,
                 init_explo=1.0, explo_strat='sample'):
        self.discrete_action = discrete_action

        if explo_strat not in ['sample', 'e_greedy']:
            print('ERROR: Bad exploration strategy with', explo_strat,
                  'given')
            exit(0)
        self.explo_strat = explo_strat

        # Networks

    def step(self, obs, explore=False, device="cpu"):
        pass

    def get_params(self):
        pass

    def load_params(self, params):
        pass


class QMIX:

    def __init__(self, n_agents, input_dim, act_dim, lr, 
                 gamma=0.99, tau=0.01, hidden_dim=64, discrete_action=False,
                 shared_params=False, init_explo_rate=1.0, 
                 explo_strat="sample"):
        self.n_agents = n_agents
        self.input_dim = input_dim
        self.act_dim = act_dim
        self.lr = lr
        self.gamma = gamma
        self.tau = tau
        self.hidden_dim = hidden_dim
        self.discrete_action = discrete_action
        self.shared_params = shared_params
        self.init_explo_rate = init_explo_rate
        self.explo_strat = explo_strat

        # Create agent policies
        if not shared_params:
            self.agents = [QMIXAgent(
                    input_dim, act_dim, lr, hidden_dim, discrete_action, 
                    init_explo_rate, explo_strat)
                for _ in range(n_agents)]
        else:
            self.agents = [QMIXAgent(
                    input_dim, act_dim, lr, hidden_dim, discrete_action, 
                    init_explo_rate, explo_strat)]

        self.device = "cpu"

    def scale_noise(self, scale):
        """
        Scale noise for each agent
        Inputs:
            scale (float): scale of noise
        """
        for a in self.agents:
            a.scale_noise(scale)

    def reset_noise(self):
        for a in self.agents:
            a.reset_noise()

    def prep_training(self, device='cpu'):
        pass

    def prep_rollouts(self, device='cpu'):
        pass

    def step(self, observations, explore=False):
        pass

    def update(self, sample, agent_i):
        pass

    def update_all_targets(self):
        pass

    def save(self, filename):
        pass