import torch
import random
import numpy as np

from torch.optim import Adam

#from .networks import MLPNetwork
from .utils import hard_update, OUNoise, gumbel_softmax, onehot_from_logits, soft_update

MSELoss = torch.nn.MSELoss()



import torch.nn as nn
import torch.nn.functional as F

class MLPNetwork(nn.Module):
    """
    MLP network (can be used as value or policy)
    """
    def __init__(self, input_dim, out_dim, hidden_dim=64, nonlin=F.relu,
                 constrain_out=False, norm_in=True, discrete_action=True):
        """
        Inputs:
            input_dim (int): Number of dimensions in input
            out_dim (int): Number of dimensions in output
            hidden_dim (int): Number of hidden dimensions
            nonlin (PyTorch function): Nonlinearity to apply to hidden layers
        """
        super(MLPNetwork, self).__init__()

        if norm_in:  # normalize inputs
            self.in_fn = nn.BatchNorm1d(input_dim)
            self.in_fn.weight.data.fill_(1)
            self.in_fn.bias.data.fill_(0)
        else:
            self.in_fn = lambda x: x
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.nonlin = nonlin
        if constrain_out and not discrete_action:
            # initialize small to prevent saturation
            self.fc3.weight.data.uniform_(-3e-3, 3e-3)
            self.out_fn = torch.tanh
        else:  # logits for discrete action (will softmax later)
            self.out_fn = lambda x: x

    def forward(self, X):
        """
        Inputs:
            X (PyTorch Matrix): Batch of observations
        Outputs:
            out (PyTorch Matrix): Output of network (actions, values, etc)
        """
        h1 = self.nonlin(self.fc1(self.in_fn(X)))
        h2 = self.nonlin(self.fc2(h1))
        out = self.out_fn(self.fc3(h2))
        return out







class DDPGAgent(object):
    """
    General class for DDPG agents (policy, critic, target policy, target
    critic, exploration noise)
    """
    def __init__(self, num_in_pol, num_out_pol, num_in_critic, hidden_dim=64,
                 lr=0.01, discrete_action=True, init_exploration=0.3,
                 exploration_strategy='sample'):
        """
        Inputs:
            num_in_pol (int): number of dimensions for policy input
            num_out_pol (int): number of dimensions for policy output
            num_in_critic (int): number of dimensions for critic input
            hidden_dim (int): number of neurons in the hidden layer of the
                policy and critic
            lr (float): learning rate
            discrete_action (boolean): if True, the model should output 
                discrete actions
            init_exploration (float): initial exploration rate (or noise if
                actions are continuous)
            exploration_strategy (str): if actions are discrete, stratefy for
                choosing actions when exploring, either 'sample' or 'e_greedy'
        """
        self.policy = MLPNetwork(num_in_pol, num_out_pol,
                                 hidden_dim=hidden_dim,
                                 constrain_out=True,
                                 discrete_action=discrete_action)
        self.critic = MLPNetwork(num_in_critic, 1,
                                 hidden_dim=hidden_dim,
                                 constrain_out=False)
        self.target_policy = MLPNetwork(num_in_pol, num_out_pol,
                                        hidden_dim=hidden_dim,
                                        constrain_out=True,
                                        discrete_action=discrete_action)
        self.target_critic = MLPNetwork(num_in_critic, 1,
                                        hidden_dim=hidden_dim,
                                        constrain_out=False)
        hard_update(self.target_policy, self.policy)
        hard_update(self.target_critic, self.critic)
        self.policy_optimizer = Adam(self.policy.parameters(), lr=lr)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=lr)
        # self.optimizer = Adam(
        #     list(self.policy.parameters()) + list(self.critic.parameters()),
        #     lr=lr)
        if not discrete_action:
            self.exploration = OUNoise(num_out_pol)
        else:
            self.exploration = init_exploration  # epsilon for eps-greedy
        self.discrete_action = discrete_action
        if exploration_strategy not in ['sample', 'e_greedy']:
            print('ERROR: Bad exploration strategy with', exploration_strategy,
                'given')
            exit(0)
        self.exploration_strategy = exploration_strategy

    def reset_noise(self):
        if not self.discrete_action:
            self.exploration.reset()

    def scale_noise(self, scale):
        if self.discrete_action:
            self.exploration = scale
        else:
            self.exploration.scale = scale

    def step(self, obs, explore=False):
        """
        Take a step forward in environment for a minibatch of observations
        Inputs:
            obs (PyTorch Variable): Observations for this agent
            explore (boolean): Whether or not to add exploration noise
        Outputs:
            action (PyTorch Variable): Actions for this agent
        """
        action = self.policy(obs)
        if self.discrete_action:
            if explore:
                if self.exploration_strategy == 'sample':
                    # Sample from probabilities given by the policy
                    action = gumbel_softmax(action, hard=True)
                elif self.exploration_strategy == 'e_greedy':
                    # Sample random number
                    r = random.uniform(0, 1)
                    # Exploration
                    if r <= self.exploration:
                        # Take random action (random one-hot vector)
                        action_dim = action.shape[1]
                        action = np.eye(action_dim)[random.randint(
                            0, action_dim - 1)]
                        action = torch.Tensor(action).unsqueeze(0)
                    # Exploitation
                    else:
                        # Take most probable action
                        action = onehot_from_logits(action)
            else:
                # Take most probable action
                action = onehot_from_logits(action)
        else:  # continuous action
            if explore:
                # Add noise to model's action
                action += torch.Tensor(self.exploration.noise())
            action = action.clamp(-1, 1)
        return action

    def get_params(self):
        return {'policy': self.policy.state_dict(),
                'critic': self.critic.state_dict(),
                'target_policy': self.target_policy.state_dict(),
                'target_critic': self.target_critic.state_dict(),
                'policy_optimizer': self.policy_optimizer.state_dict(),
                'critic_optimizer': self.critic_optimizer.state_dict()}

    def load_params(self, params):
        self.policy.load_state_dict(params['policy'])
        self.critic.load_state_dict(params['critic'])
        self.target_policy.load_state_dict(params['target_policy'])
        self.target_critic.load_state_dict(params['target_critic'])
        self.policy_optimizer.load_state_dict(params['policy_optimizer'])
        self.critic_optimizer.load_state_dict(params['critic_optimizer'])
        



class MADDPG(object):
    """
    Wrapper class for DDPG-esque (i.e. also MADDPG) agents in multi-agent task
    """
    def __init__(self, n_agents, input_dim, act_dim, gamma=0.95, tau=0.01, 
                 lr=0.01, hidden_dim=64, discrete_action=False, 
                 shared_params=False, init_explo=0.3, 
                 exploration_strategy='sample'):
        """
        Inputs:
            agent_init_params (list of dict): List of dicts with parameters to
                                              initialize each agent
                num_in_pol (int): Input dimensions to policy
                num_out_pol (int): Output dimensions to policy
                num_in_critic (int): Input dimensions to critic
            alg_types (list of str): Learning algorithm for each agent (DDPG
                                       or MADDPG)
            gamma (float): Discount factor
            tau (float): Target update rate
            lr (float): Learning rate for policy and critic
            hidden_dim (int): Number of hidden dimensions for networks
            discrete_action (bool): Whether or not to use discrete action space
        """
        self.nagents = n_agents
        self.alg_types = ['MADDPG', 'MADDPG']
        self.shared_params = shared_params
        agent_init_params = {
            "num_in_pol": input_dim,
            "num_out_pol": act_dim,
            "num_in_critic": n_agents * input_dim + n_agents * act_dim
        }
        agent_init_params = [agent_init_params, agent_init_params]
        if not shared_params:
            self.agents = [DDPGAgent(lr=lr, discrete_action=discrete_action,
                                    hidden_dim=hidden_dim, 
                                    init_exploration=init_explo,
                                    exploration_strategy=exploration_strategy,
                                    **params)
                            for params in agent_init_params]
        else:
            model = DDPGAgent(lr=lr, discrete_action=discrete_action,
                              hidden_dim=hidden_dim, 
                              init_exploration=init_explo,
                              exploration_strategy=exploration_strategy,
                              **agent_init_params[0])
            self.agents = [model for _ in agent_init_params]
        self.agent_init_params = agent_init_params
        self.gamma = gamma
        self.tau = tau
        self.lr = lr
        self.discrete_action = discrete_action
        self.pol_dev = 'cpu'  # device for policies
        self.critic_dev = 'cpu'  # device for critics
        self.trgt_pol_dev = 'cpu'  # device for target policies
        self.trgt_critic_dev = 'cpu'  # device for target critics
        self.niter = 0

    @property
    def policies(self):
        return [a.policy for a in self.agents]

    @property
    def target_policies(self):
        return [a.target_policy for a in self.agents]

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

    def step(self, observations, explore=False):
        """
        Take a step forward in environment with all agents
        Inputs:
            observations: List of observations for each agent
            explore (boolean): Whether or not to add exploration noise
        Outputs:
            actions: List of actions for each agent
        """
        return [a.step(obs.unsqueeze(0), explore=explore)
                for a, obs in zip(self.agents, observations)]

    def update(self, sample, agent_i, parallel=False, logger=None):
        """
        Update parameters of agent model based on sample from replay buffer
        Inputs:
            sample: tuple of (observations, actions, rewards, next
                    observations, and episode end masks) sampled randomly from
                    the replay buffer. Each is a list with entries
                    corresponding to each agent
            agent_i (int): index of agent to update
            parallel (bool): If true, will average gradients across threads
            logger (SummaryWriter from Tensorboard-Pytorch):
                If passed in, important quantities will be logged
        """
        obs, acs, rews, next_obs, dones = sample
        curr_agent = self.agents[agent_i]
        # curr_agent.optimizer.zero_grad()
        # Critic Update
        curr_agent.critic_optimizer.zero_grad()
        # Compute Target Value
        if self.alg_types[agent_i] == 'MADDPG':
            if self.discrete_action: # one-hot encode action
                all_trgt_acs = [onehot_from_logits(pi(nobs)) for pi, nobs in
                                zip(self.target_policies, next_obs)]
            else:
                all_trgt_acs = [pi(nobs) for pi, nobs in zip(self.target_policies,
                                                             next_obs)]
            trgt_vf_in = torch.cat((*next_obs, *all_trgt_acs), dim=1)
        else:  # DDPG
            if self.discrete_action:
                trgt_vf_in = torch.cat((next_obs[agent_i],
                                        onehot_from_logits(
                                            curr_agent.target_policy(
                                                next_obs[agent_i]))),
                                       dim=1)
            else:
                trgt_vf_in = torch.cat((next_obs[agent_i],
                                        curr_agent.target_policy(next_obs[agent_i])),
                                       dim=1)
        target_value = (rews[agent_i].view(-1, 1) + self.gamma *
                        curr_agent.target_critic(trgt_vf_in) *
                        (1 - dones[agent_i].view(-1, 1)))
        # Compute Value
        if self.alg_types[agent_i] == 'MADDPG':
            vf_in = torch.cat((*obs, *acs), dim=1)
        else:  # DDPG
            vf_in = torch.cat((obs[agent_i], acs[agent_i]), dim=1)
        actual_value = curr_agent.critic(vf_in)
        # Value loss = minimise TD error (difference between target and value)
        vf_loss = MSELoss(actual_value, target_value.detach())
        vf_loss.backward()
        torch.nn.utils.clip_grad_norm_(curr_agent.critic.parameters(), 0.5)
        curr_agent.critic_optimizer.step()

        # Policy Update
        curr_agent.policy_optimizer.zero_grad()
        # Get Action
        if self.discrete_action:
            # Forward pass as if onehot (hard=True) but backprop through a differentiable
            # Gumbel-Softmax sample. The MADDPG paper uses the Gumbel-Softmax trick to backprop
            # through discrete categorical samples, but I'm not sure if that is
            # correct since it removes the assumption of a deterministic policy for
            # DDPG. Regardless, discrete policies don't seem to learn properly without it.
            curr_pol_out = curr_agent.policy(obs[agent_i])
            curr_pol_vf_in = gumbel_softmax(curr_pol_out, hard=True)
        else:
            curr_pol_out = curr_agent.policy(obs[agent_i])
            curr_pol_vf_in = curr_pol_out
        if self.alg_types[agent_i] == 'MADDPG':
            all_pol_acs = []
            for i, pi, ob in zip(range(self.nagents), self.policies, obs):
                if i == agent_i:
                    all_pol_acs.append(curr_pol_vf_in)
                elif self.discrete_action:
                    all_pol_acs.append(onehot_from_logits(pi(ob)))
                else:
                    all_pol_acs.append(pi(ob).detach())
            vf_in = torch.cat((*obs, *all_pol_acs), dim=1)
        else:  # DDPG
            vf_in = torch.cat((obs[agent_i], curr_pol_vf_in),
                              dim=1)
        # Policy loss = maximise value of our actions
        pol_loss = -curr_agent.critic(vf_in).mean()
        pol_loss += (curr_pol_out**2).mean() * 1e-3
        pol_loss.backward()
        torch.nn.utils.clip_grad_norm_(curr_agent.policy.parameters(), 0.5)
        curr_agent.policy_optimizer.step()
        # curr_agent.optimizer.step()
        if logger is not None:
            logger.add_scalars('agent%i/losses' % agent_i,
                               {'vf_loss': vf_loss,
                                'pol_loss': pol_loss},
                               self.niter)

    def update_all_targets(self):
        """
        Update all target networks (called after normal updates have been
        performed for each agent)
        """
        if self.shared_params:
            soft_update(self.agents[0].target_critic, self.agents[0].critic, self.tau)
            soft_update(self.agents[0].target_policy, self.agents[0].policy, self.tau)
        else:
            for a in self.agents:
                soft_update(a.target_critic, a.critic, self.tau)
                soft_update(a.target_policy, a.policy, self.tau)
        self.niter += 1

    def prep_training(self, device='cpu'):
        for a in self.agents:
            a.policy.train()
            a.critic.train()
            a.target_policy.train()
            a.target_critic.train()
        if type(device) is str:
            device = torch.device(device)
        fn = lambda x: x.to(device)
        if not self.pol_dev == device:
            for a in self.agents:
                a.policy = fn(a.policy)
            self.pol_dev = device
        if not self.critic_dev == device:
            for a in self.agents:
                a.critic = fn(a.critic)
            self.critic_dev = device
        if not self.trgt_pol_dev == device:
            for a in self.agents:
                a.target_policy = fn(a.target_policy)
            self.trgt_pol_dev = device
        if not self.trgt_critic_dev == device:
            for a in self.agents:
                a.target_critic = fn(a.target_critic)
            self.trgt_critic_dev = device

    def prep_rollouts(self, device='cpu'):
        for a in self.agents:
            a.policy.eval()
        if type(device) is str:
            device = torch.device(device)
        fn = lambda x: x.to(device)
        # only need main policy for rollouts
        if not self.pol_dev == device:
            for a in self.agents:
                a.policy = fn(a.policy)
            self.pol_dev = device

    def save(self, filename):
        """
        Save trained parameters of all agents into one file
        """
        self.prep_training(device='cpu')  # move parameters to CPU before saving
        save_dict = {'agent_params': [a.get_params() for a in self.agents]}
        torch.save(save_dict, filename)

    def load_cp(self, cp_path):
        save_dict = torch.load(cp_path, map_location=torch.device('cpu'))
        for a, params in zip(self.agents, save_dict['agent_params']):
            a.load_params(params)

    @classmethod
    def init_from_save(cls, filename):
        """
        Instantiate instance of this class from file created by 'save' method
        """
        save_dict = torch.load(filename, map_location=torch.device('cpu'))
        instance = cls(**save_dict['init_dict'])
        instance.init_dict = save_dict['init_dict']
        for a, params in zip(instance.agents, save_dict['agent_params']):
            a.load_params(params)
        return instance