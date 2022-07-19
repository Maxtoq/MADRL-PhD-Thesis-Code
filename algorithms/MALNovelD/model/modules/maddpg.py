import torch
import random
import numpy as np

from .networks import MLPNetwork
from .utils import hard_update, OUNoise, gumbel_softmax, onehot_from_logits


class DDPGAgent:

    def __init__(self, policy_in_dim, policy_out_dim, critic_in_dim, 
                 hidden_dim=64, init_explo=1.0, discrete_action=False,
                 explo_strat='sample'):
        self.discrete_action = discrete_action
        if explo_strat not in ['sample', 'e_greedy']:
            print('ERROR: Bad exploration strategy with', explo_strat,
                'given')
            exit(0)
        self.explo_strat = explo_strat

        # Networks
        self.policy = MLPNetwork(policy_in_dim, policy_out_dim,
                                 hidden_dim=hidden_dim)
        self.critic = MLPNetwork(critic_in_dim, 1, hidden_dim=hidden_dim)
        # Target networks
        self.target_policy = MLPNetwork(policy_in_dim, policy_out_dim,
                                        hidden_dim=hidden_dim)
        self.target_critic = MLPNetwork(critic_in_dim, 1, 
                                        hidden_dim=hidden_dim)
        # Copy parameters in targets
        hard_update(self.target_policy, self.policy)
        hard_update(self.target_critic, self.critic)

        # Exploration
        if not discrete_action:
            # Noise added to action
            self.exploration = OUNoise(policy_out_dim)
        else:
            # Epsilon-greedy
            self.exploration = init_explo

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
                if self.explo_strat == 'sample':
                    # Sample from probabilities given by the policy
                    action = gumbel_softmax(action, hard=True)
                elif self.explo_strat == 'e_greedy':
                    # Sample random number
                    r = random.uniform(0, 1)
                    # Exploration
                    if r <= self.exploration:
                        # Take random action (random one-hot vector)
                        action_dim = action.shape[1]
                        action = np.eye(action_dim)[random.randint(
                            0, action_dim - 1)]
                        action = torch.Variable(
                            torch.Tensor(action).unsqueeze(0),
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
                action += torch.Variable(
                    torch.Tensor(self.exploration.noise()),
                    requires_grad=False)
            action = action.clamp(-1, 1)
        return action

    def get_params(self):
        return {'policy': self.policy.state_dict(),
                'critic': self.critic.state_dict(),
                'target_policy': self.target_policy.state_dict(),
                'target_critic': self.target_critic.state_dict()}

    def load_params(self, params):
        self.policy.load_state_dict(params['policy'])
        self.critic.load_state_dict(params['critic'])
        self.target_policy.load_state_dict(params['target_policy'])
        self.target_critic.load_state_dict(params['target_critic'])


class MADDPG:

    def __init__(self, n_agents, obs_dim, act_dim, hidden_dim=64, 
                 discrete_action=False, shared_params=False):
        self.n_agents = n_agents
        self.shared_params = shared_params
        self.discrete_action = discrete_action

        # Create agent models
        if not shared_params:
            self.agents = [DDPGAgent()]