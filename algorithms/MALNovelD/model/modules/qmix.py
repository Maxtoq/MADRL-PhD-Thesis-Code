import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.distributions import Categorical

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
                dim=(seq_len, batch_size, obs_dim).
            rnn_states (torch.Tensor): Hidden states with which to initialise
                the RNN, dim=(1, batch_size, hidden_dim).
        Outputs:
            q_outs (torch.Tensor): Q-values for every action, 
                dim=(seq_len, batch_size, act_dim).
            new_rnn_states (torch.Tensor): Final hidden states of the RNN, 
                dim=(1, batch_size, hidden_dim).
        """
        rnn_in = self.mlp_in(obs)

        rnn_outs, new_rnn_states = self.rnn(rnn_in, rnn_states)

        q_outs = self.mlp_out(rnn_outs)

        return q_outs, new_rnn_states


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

    def __init__(self, q_in_dim, q_out_dim, lr, 
                 hidden_dim=64, init_explo=1.0):
        self.epsilon = init_explo
        self.q_out_dim = q_out_dim

        # Q function
        self.q_network = DRQNetwork(q_in_dim, q_out_dim, hidden_dim)

    def set_explo_rate(self, explo_rate):
        self.epsilon = explo_rate

    def get_q_values(self, obs, last_acts, qnet_rnn_states):
        """
        Returns Q-values computes from given inputs.
        Inputs:
            obs (torch.Tensor): Agent's observation batch, dim=([seq_len], 
                batch_size, obs_dim).
            last_acts (torch.Tensor): Agent's last action batch, 
                dim=([seq_len], batch_size, act_dim).
            qnet_rnn_states (torch.Tensor): Agents' Q-network hidden states
                batch, dim=(1, batch_size, hidden_dim).
        Output:
            q_values (torch.Tensor): Q_values, dim=([seq_len], batch_size, act_dim).
            new_qnet_rnn_states (torch.Tensor): New hidden states of the 
                Q-network, dim=(1, batch_size, hidden_dim).
        """
        # Concatenate observation and last actions
        qnet_input = torch.cat((obs, last_acts), dim=-1)

        # Get Q-values
        q_values, new_qnet_rnn_states = self.q_network(
            qnet_input, qnet_rnn_states)

        return q_values, new_qnet_rnn_states

    def get_actions(self, obs, last_acts, qnet_rnn_states, explore=False):
        """
        Returns an action chosen using the Q-network.
        Inputs:
            obs (torch.Tensor): Agent's observation batch, dim=([seq_len], 
                batch_size, obs_dim).
            last_acts (torch.Tensor): Agent's last action batch, 
                dim=([seq_len], batch_size, act_dim).
            qnet_rnn_states (torch.Tensor): Agents' Q-network hidden states
                batch, dim=(1, batch_size, hidden_dim).
            explore (bool): Whether to perform exploration or exploitation.
        Output:
            onehot_actions (torch.Tensor): Chosen actions, dim=([seq_len], 
                batch_size, act_dim).
            greedy_Qs (torch.Tensor): Q-values corresponding to greedy actions,
                dim=([seq_len], batch_size, 1).
            new_qnet_rnn_states (torch.Tensor): New agent's Q-network hidden 
                states dim=(1, batch_size, hidden_dim).
        """
        # Compute Q-values
        q_values, new_qnet_rnn_states = self.get_q_values(
            obs, last_acts, qnet_rnn_states)

        batch_size = obs.shape[-2]
        # Choose actions
        greedy_Qs, greedy_actions = q_values.max(dim=-1)
        if explore:
            # Sample random number for each action
            rands = torch.rand(batch_size)
            take_random = (rands < self.epsilon).int()
            # Get random actions
            rand_actions = Categorical(
                logits=torch.ones(batch_size, self.q_out_dim)).sample()
            # Choose actions
            actions = (1 - take_random) * greedy_actions + \
                      take_random * rand_actions
            onehot_actions = torch.eye(self.q_out_dim)[actions]
        else:
            onehot_actions = torch.eye(self.q_out_dim)[greedy_actions]
        
        return onehot_actions, greedy_Qs, new_qnet_rnn_states

    def get_params(self):
        pass

    def load_params(self, params):
        pass


class QMIX:

    def __init__(self, n_agents, obs_dim, act_dim, lr, 
                 gamma=0.99, tau=0.01, hidden_dim=64, shared_params=False, 
                 init_explo_rate=1.0, device="cpu"):
        self.n_agents = n_agents
        self.input_dim = obs_dim
        self.act_dim = act_dim
        self.lr = lr
        self.gamma = gamma
        self.tau = tau
        self.hidden_dim = hidden_dim
        self.shared_params = shared_params
        self.init_explo_rate = init_explo_rate

        # Create agent policies
        if not shared_params:
            self.agents = [QMIXAgent(
                    obs_dim, act_dim, lr, hidden_dim, init_explo_rate)
                for _ in range(n_agents)]
            self.last_actions = [
                torch.zeros(1, act_dim, device=device)] * self.n_agents
            self.qnets_hidden_states = [
                torch.zeros((1, 1, act_dim), device=device)] * self.n_agents
        else:
            self.agents = [QMIXAgent(
                    obs_dim, act_dim, lr, hidden_dim, init_explo_rate)]
            self.last_actions = torch.zeros((self.n_agents, act_dim), device=device)
            self.qnets_hidden_states = torch.zeros(
                (1, self.n_agents, act_dim), device=device)

        self.device = device

    def set_explo_rate(self, explo_rate):
        """
        Set exploration rate for each agent
        Inputs:
            explo_rate (float): New exploration rate.
        """
        for a in self.agents:
            a.set_explo_rate(explo_rate)

    def get_actions(self, obs_list, explore=False):
        """
        Returns each agent's action given their observation.
        Inputs:
            obs_list (list(numpy.ndarray)): List of agent observations.
            explore (bool): Whether to explore or not.
        Outputs:
            actions (list(torch.Tensor)): Each agent's chosen action.
        """
        if self.shared_params:
            obs = torch.Tensor(np.array(obs_list)).to(self.device)
            actions, _, new_qnets_hidden_states = self.agents[0].get_actions(
                obs, self.last_actions, self.qnets_hidden_states)
            actions = [actions[a_i] for a_i in range(self.n_agents)]
            self.last_actions = actions
            self.qnets_hidden_states = new_qnets_hidden_states
        else:
            actions = [
                agent.get_actions(torch.Tensor(obs)) 
                for agent, obs in zip(self.agents, obs_list)]
        return actions

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