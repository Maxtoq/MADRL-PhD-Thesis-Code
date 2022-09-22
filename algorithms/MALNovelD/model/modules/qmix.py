import copy
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

    def __init__(self, n_agents, input_dim,
            mixer_hidden_dim=32, hypernet_hidden_dim=64, device="cpu"):
        super(QMixer, self).__init__()
        self.n_agents = n_agents
        self.input_dim = input_dim
        self.device = device
        self.mixer_hidden_dim = mixer_hidden_dim
        self.hypernet_hidden_dim = hypernet_hidden_dim

        # Hypernets
        # self.hypernet_weights1 = MLPNetwork(
        #     input_dim, n_agents * mixer_hidden_dim, hypernet_hidden_dim, 0)
        self.hypernet_weights1 = get_init_linear(
            input_dim, n_agents * mixer_hidden_dim).to(device)
        self.hypernet_bias1 = get_init_linear(
            input_dim, mixer_hidden_dim).to(device)
        # self.hypernet_weights2 = MLPNetwork(
        #     input_dim, mixer_hidden_dim, hypernet_hidden_dim, 0)
        self.hypernet_weights2 = get_init_linear(
            input_dim, mixer_hidden_dim).to(device)
        self.hypernet_bias2 = MLPNetwork(
            input_dim, 1, hypernet_hidden_dim, 0).to(device)

    def forward(self, local_qs, obs):
        """
        Computes Q_tot using local agent q-values and global observation.
        Inputs:
            local_qs (torch.Tensor): Local agent q-values, dim=(episode_length, 
                batch_size, n_agents).
            obs (torch.Tensor): Global observation, i.e. concatenated local 
                observations, dimension=(episode_lenght, batch_size, 
                n_agents * (obs_dim + act_dim))
        Outputs:
            Q_tot (torch.Tensor): Global Q-value computed by the mixer, 
                dim=(episode_length, batch_size, 1, 1).
        """
        batch_size = local_qs.size(1)
        obs = obs.view(-1, batch_size, self.input_dim).float()
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

    def __init__(self, q_in_dim, q_out_dim, 
                 hidden_dim=64, init_explo=1.0, device="cpu"):
        self.epsilon = init_explo
        self.q_out_dim = q_out_dim
        self.hidden_dim = hidden_dim

        # Q function
        self.q_net = DRQNetwork(q_in_dim, q_out_dim, hidden_dim).to(device)
        # Target Q function
        self.target_q_net = copy.deepcopy(self.q_net)

    def set_explo_rate(self, explo_rate):
        self.epsilon = explo_rate

    def get_init_hidden(self, batch_size, device="cpu"):
        """
        Returns a zero tensor for initialising the hidden state of the 
        Q-network.
        Inputs:
            batch_size (int): Batch size needed for the tensor.
            device (str): CUDA device to put the tensor on.
        Outputs:
            init_hidden (torch.Tensor): Batch of zero-filled hidden states,
                dim=(1, batch_size, hidden_dim).
        """
        return torch.zeros((1, batch_size, self.hidden_dim), device=device)

    def q_values_from_actions(self, q_batch, action_batch):
        """
        Get Q-values corresponding to actions.
        Inputs:
            q_batch (torch.Tensor): Batch of Q-values, dim=(seq_len, 
                batch_size, act_dim).
            action_batch (torch.Tensor): Batch of one-hot actions taken by the
                agent, dim=(seq_len, batch_size, act_dim).
        Output:
            q_values (torch.Tensor): Q-values in q_batch corresponding to 
                actions in action_batch, dim=(seq_len, batch_size, 1).
        """
        # Convert one-hot actions to index
        action_ids = action_batch.max(dim=-1)
        print(action_ids.shape)

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
        # Check if input is a sequence of observations
        no_seq = len(obs.shape) == 2

        # Concatenate observation and last actions
        qnet_input = torch.cat((obs, last_acts), dim=-1)

        if no_seq:
            qnet_input = qnet_input.unsqueeze(0)

        # Get Q-values
        q_values, new_qnet_rnn_states = self.q_net(qnet_input, qnet_rnn_states)

        if no_seq:
            q_values = q_values.squeeze(0)

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
                dim=([seq_len], batch_size).
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

    def __init__(self, nb_agents, obs_dim, act_dim, lr, 
                 gamma=0.99, tau=0.01, hidden_dim=64, shared_params=False, 
                 init_explo_rate=1.0, max_grad_norm=None, device="cpu"):
        self.nb_agents = nb_agents
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.lr = lr
        self.gamma = gamma
        self.tau = tau
        self.hidden_dim = hidden_dim
        self.shared_params = shared_params
        self.init_explo_rate = init_explo_rate
        self.max_grad_norm = max_grad_norm
        self.device = device

        # Create agent policies
        if not shared_params:
            self.agents = [QMIXAgent(
                    obs_dim + act_dim, 
                    act_dim,
                    hidden_dim, 
                    init_explo_rate,
                    device)
                for _ in range(nb_agents)]
        else:
            self.agents = [QMIXAgent(
                    obs_dim + act_dim, 
                    act_dim,
                    hidden_dim, 
                    init_explo_rate,
                    device)]
        # Initialise last actions and Q-networks hidden states
        self.last_actions = None
        self.qnets_hidden_states = None
        self.reset_new_episode()

        # Create Q-mixer network
        mixer_in_dim = nb_agents * (obs_dim + act_dim)
        self.mixer = QMixer(nb_agents, mixer_in_dim, device=device)
        # Target Q-mixer
        self.target_mixer = copy.deepcopy(self.mixer)

        # Initiate optimiser with all parameters
        self.parameters = []
        for ag in self.agents:
            self.parameters += ag.q_net.parameters()
        self.parameters += self.mixer.parameters()
        self.optimizer = torch.optim.RMSprop(self.parameters, lr)

    def reset_new_episode(self):
        """ 
        Initialises last actions and Q-network hidden states tensor with 
        zero-filled tensors.
        """
        if not self.shared_params:
            self.last_actions = [
                torch.zeros(1, self.act_dim, device=self.device)
            ] * self.nb_agents
            self.qnets_hidden_states = [
                self.agents[0].get_init_hidden(1, self.device)
            ] * self.nb_agents
        else:
            self.last_actions = torch.zeros(
                (self.nb_agents, self.act_dim), device=self.device)
            self.qnets_hidden_states = self.agents[0].get_init_hidden(
                self.nb_agents, self.device)

    def set_explo_rate(self, explo_rate):
        """
        Set exploration rate for each agent
        Inputs:
            explo_rate (float): New exploration rate.
        """
        for a in self.agents:
            a.set_explo_rate(explo_rate)

    def prep_training(self, device='cpu'):
        for a in self.agents:
            a.q_net.train()
            a.q_net = a.q_net.to(device)
        self.device = device

    def prep_rollouts(self, device='cpu'):
        for a in self.agents:
            a.q_net.eval()
            a.q_net = a.q_net.to(device)
        self.device = device

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
            actions_batch, _, new_qnets_hidden_states = self.agents[0].get_actions(
                obs, self.last_actions, self.qnets_hidden_states, explore)
            actions = [actions_batch[a_i] for a_i in range(self.nb_agents)]
            self.last_actions = actions_batch
            self.qnets_hidden_states = new_qnets_hidden_states
        else:
            actions = []
            for a_i in range(self.nb_agents):
                obs = torch.Tensor(obs_list[a_i]).unsqueeze(0).to(self.device)
                action, _, new_qnet_hidden_state = self.agents[a_i].get_actions(
                    obs, 
                    self.last_actions[a_i], 
                    self.qnets_hidden_states[a_i],
                    explore
                )
                actions.append(action.squeeze())
                self.last_actions[a_i] = action
                self.qnets_hidden_states[a_i] = new_qnet_hidden_state
        return actions

    def train_on_batch(self, batch):
        obs_b, shared_obs_b, act_b, rew_b, done_b = batch

        batch_size = obs_b.shape[2]

        agent_qs = []
        agent_nqs = []
        for a_i in range(self.nb_agents):
            agent = self.agents[0] if self.shared_params else self.agents[a_i]

            obs_ag = obs_b[a_i]
            shared_obs_ag = shared_obs_b[a_i]
            act_ag = act_b[a_i]
            rew_ag = rew_b[a_i]
            done_ag = done_b[a_i]

            prev_act_ag = torch.cat((
                torch.zeros(1, batch_size, self.act_dim).to(self.device),
                act_ag
            ))

            q_values, _ = agent.get_q_values(
                obs_ag, 
                prev_act_ag, 
                agent.get_init_hidden(batch_size, self.device)
            )
            
            actions_q_values = agent.q_values_from_actions(q_values, act_ag)


    def update_all_targets(self):
        pass

    def save(self, filename):
        pass