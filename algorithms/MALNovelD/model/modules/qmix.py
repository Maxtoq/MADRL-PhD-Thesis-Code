import torch
import torch.nn as nn
import torch.nn.functional as F

from .networks import MLPNetwork, get_init_linear


class DRQNetwork(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_dim=64):
        super(DRQNetwork, self).__init__()
        
        self.mlp_in = get_init_linear(input_dim, hidden_dim)

        self.rnn = nn.GRU(hidden_dim, hidden_dim)

        self.mlp_out = get_init_linear(hidden_dim, output_dim)

    def forward(self, obs, rnn_states):
        """
        Compute q values for every action given observations and rnn states.
        Inputs:
            obs (torch.Tensor): Observations from which to compute q-values,
                dim=(batch_size, obs_dim).
            rnn_states (torch.Tensor): Hidden states with which to initialise
                the RNN, dim=(batch_size, hidden_dim).
        Outputs:
            q_outs (torch.Tensor): Q-values for every action, 
                dim=(batch_size, act_dim).
            h_final (torch.Tensor): Final hidden states of the RNN, 
                dim=(batch_size, hidden_dim).
        """


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
        # self.hypernet_weights1 = MLPNetwork(
        #     cent_obs_dim, n_agents * mixer_hidden_dim, hypernet_hidden_dim, 0)
        self.hypernet_weights1 = get_init_linear(
            cent_obs_dim, n_agents * mixer_hidden_dim)
        self.hypernet_bias1 = get_init_linear(cent_obs_dim, mixer_hidden_dim)
        # self.hypernet_weights2 = MLPNetwork(
        #     cent_obs_dim, mixer_hidden_dim, hypernet_hidden_dim, 0)
        self.hypernet_weights2 = get_init_linear(
            cent_obs_dim, mixer_hidden_dim)
        self.hypernet_bias2 = MLPNetwork(
            cent_obs_dim, 1, hypernet_hidden_dim, 0)

    def forward(self, local_qs, obs):
        """
        Computes Q_tot using local agent q-values and global observation.
        Inputs:
            local_qs (torch.Tensor): Local agent q-values, dim=(episode_length, 
                batch_size, n_agents).
            obs (torch.Tensor): Global observation, i.e. concatenated local 
                observations, dimension=(episode_lenght, batch_size, 
                n_agents * obs_dim)
        Outputs:
            Q_tot (torch.Tensor): Global Q-value computed by the mixer, 
                dim=(episode_length, batch_size, 1, 1).
        """
        batch_size = local_qs.size(1)
        obs = obs.view(-1, batch_size, self.cent_obs_dim).float()
        local_qs = local_qs.view(-1, batch_size, 1, self.num_mixer_q_inps)

        # First layer forward pass
        w1 = torch.abs(self.hypernet_weights1(obs))
        b1 = self.hypernet_bias1(obs)
        w1 = w1.view(-1, batch_size, self.n_agents, self.mixer_hidden_dim)
        b1 = b1.view(-1, batch_size, 1, self.mixer_hidden_dim)
        hidden_layer = F.elu(torch.matmul(local_qs, w1) + b1)

        # Second layer forward pass
        w2 = torch.abs(self.hyper_w2(obs))
        b2 = self.hyper_b2(obs)
        w2 = w2.view(-1, batch_size, self.mixer_hidden_dim, 1)
        b2 = b2.view(-1, batch_size, 1, 1)
        out = torch.matmul(hidden_layer, w2) + b2
        q_tot = out.view(-1, batch_size, 1, 1)

        return q_tot


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

        # Q function
        self.q_network = DRQNetwork()

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