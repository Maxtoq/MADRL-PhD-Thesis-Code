import random
import torch

import numpy as np

from torch import Tensor
from torch.autograd import Variable
from torch.optim import Adam
from .networks import MLPNetwork
from .misc import hard_update, gumbel_softmax, onehot_from_logits
from .noise import OUNoise

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
                        action = Variable(Tensor(action).unsqueeze(0),
                            requires_grad=False)
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
                action += Variable(Tensor(self.exploration.noise()),
                                   requires_grad=False)
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


class DDPGCommAgent(object):

    def __init__(self, dim_obs, dim_act, dim_in_critic, 
                 dim_com, hidden_dim=64,
                 lr=0.01, discrete_action=True):
        """
        Inputs:
            dim_obs (int): dimension of observations
            dim_act (int): dimension of actions
            dim_in_critic (int): dimension of critic input
            dim_com (int): dimension of communication act
        """
        # Action 
        self.ddpg = DDPGAgent(dim_obs, dim_act, dim_in_critic, hidden_dim, 
                              lr, discrete_action)
        # Memory
        self.mem = torch.nn.LSTM(dim_obs, hidden_dim)
        # Comm 
        self.comm = torch.nn.LSTM(hidden_dim + dim_act, dim_com)
        
