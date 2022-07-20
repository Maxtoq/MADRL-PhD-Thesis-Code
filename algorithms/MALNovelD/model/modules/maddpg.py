import torch
import random
import numpy as np

from torch.optim import Adam

from .networks import MLPNetwork
from .utils import hard_update, OUNoise, gumbel_softmax, onehot_from_logits, soft_update

MSELoss = torch.nn.MSELoss()


class DDPGAgent:

    def __init__(self, policy_in_dim, policy_out_dim, critic_in_dim, 
                 lr, hidden_dim=64, discrete_action=False, init_explo=1.0,
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

        # Optimizer
        self.optimizer = Adam(
            list(self.policy.parameters()) + list(self.critic.parameters()),
            lr=lr)

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
                action += torch.Tensor(self.exploration.noise())
            action = action.clamp(-1, 1)
        return action

    def get_params(self):
        return {'policy': self.policy.state_dict(),
                'critic': self.critic.state_dict(),
                'target_policy': self.target_policy.state_dict(),
                'target_critic': self.target_critic.state_dict(),
                'optimizer': self.optimizer.state_dict()}

    def load_params(self, params):
        self.policy.load_state_dict(params['policy'])
        self.critic.load_state_dict(params['critic'])
        self.target_policy.load_state_dict(params['target_policy'])
        self.target_critic.load_state_dict(params['target_critic'])
        self.optimizer.load_state_dict(params['optimizer'])


class MADDPG:

    def __init__(self, n_agents, input_dim, act_dim, lr=0.0007, gamma=0.95,
                 tau=0.01, hidden_dim=64, discrete_action=False, 
                 shared_params=False, init_explo_rate=1.0, 
                 explo_strat="sample"):
        self.n_agents = n_agents
        self.input_dim = input_dim
        self.act_dim = act_dim
        self.gamma = gamma
        self.tau = tau
        self.hidden_dim = hidden_dim
        self.shared_params = shared_params
        self.discrete_action = discrete_action
        self.init_explo_rate = init_explo_rate
        self.explo_strat = explo_strat

        # Create agent models
        critic_input_dim = n_agents * input_dim + n_agents * act_dim
        if not shared_params:
            self.agents = [DDPGAgent(
                    input_dim, act_dim, critic_input_dim, 
                    lr, hidden_dim, discrete_action, 
                    init_explo_rate, explo_strat)
                for _ in range(n_agents)]
        else:
            self.agents = [DDPGAgent(
                input_dim, act_dim, critic_input_dim, 
                lr, hidden_dim, discrete_action, 
                init_explo_rate, explo_strat)]

    @property
    def policies(self):
        if self.shared_params:
            return [self.agents[0].policy for _ in range(self.n_agents)]
        else:
            return [a.policy for a in self.agents]

    @property
    def target_policies(self):
        if self.shared_params:
            return [self.agents[0].target_policy for _ in range(self.n_agents)]
        else:
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

    def prep_training(self, device='cpu'):
        if type(device) is str:
            device = torch.device(device)
        for a in self.agents:
            a.policy.train()
            a.policy = a.policy.to(device)
            a.critic.train()
            a.critic = a.critic.to(device)
            a.target_policy.train()
            a.target_policy = a.target_policy.to(device)
            a.target_critic.train()
            a.target_critic = a.target_critic.to(device)

    def prep_rollouts(self, device='cpu'):
        if type(device) is str:
            device = torch.device(device)
        for a in self.agents:
            a.policy.eval()
            a.policy = a.policy.to(device)

    def step(self, observations, explore=False):
        """
        Take a step forward in environment with all agents
        Inputs:
            observations (torch.Tensor): Tensor containing observations of all
                agents, dim=(n_agents, input_dim).
            explore (boolean): Whether or not to perform exploration.
        Outputs:
            actions (list(torch.Tensor)): List of actions for each agent
        """
        if self.shared_params:
            actions_tensor = self.agents[0].step(observations, explore=explore)
            actions = list(actions_tensor)
        else:
            actions = [
                self.agents[a_i].step(observations[a_i].unsqueeze(0), 
                    explore=explore)
                for a_i in range(self.n_agents)]
        return actions

    def update(self, sample, agent_i):
        """
        Update parameters of agent model based on sample from replay buffer
        Inputs:
            sample: tuple of (observations, actions, rewards, next
                    observations, and episode end masks) sampled randomly from
                    the replay buffer. Each is a list with entries
                    corresponding to each agent.
            agent_i (int): index of agent to update.
        """
        obs, acs, rews, next_obs, dones = sample
        curr_agent = self.agents[agent_i]
        curr_agent.optimizer.zero_grad()
        # Critic Loss
        # Compute Target Value
        if self.discrete_action: # one-hot encode action
            all_trgt_acs = [onehot_from_logits(pi(nobs)) 
                for pi, nobs in zip(self.target_policies, next_obs)]
        else:
            all_trgt_acs = [pi(nobs) 
                for pi, nobs in zip(self.target_policies, next_obs)]
        trgt_vf_in = torch.cat((*next_obs, *all_trgt_acs), dim=1)
        
        target_value = (rews[agent_i].view(-1, 1) + self.gamma *
                        curr_agent.target_critic(trgt_vf_in) *
                        (1 - dones[agent_i].view(-1, 1)))
        # Compute Value
        vf_in = torch.cat((*obs, *acs), dim=1)
        actual_value = curr_agent.critic(vf_in)
        # Value loss = minimise TD error (difference between target and value)
        vf_loss = MSELoss(actual_value, target_value.detach())

        # Policy Update
        # Get Action
        curr_pol_out = curr_agent.policy(obs[agent_i])
        if self.discrete_action:
            curr_pol_vf_in = gumbel_softmax(curr_pol_out, hard=True)
        else:
            curr_pol_vf_in = curr_pol_out
        all_pol_acs = []
        for i, pi, ob in zip(range(self.n_agents), self.policies, obs):
            if i == agent_i:
                all_pol_acs.append(curr_pol_vf_in)
            elif self.discrete_action:
                all_pol_acs.append(onehot_from_logits(pi(ob)))
            else:
                all_pol_acs.append(pi(ob).detach())
        vf_in = torch.cat((*obs, *all_pol_acs), dim=1)
        # Policy loss = maximise value of our actions
        pol_loss = -curr_agent.critic(vf_in).mean()
        pol_loss += (curr_pol_out**2).mean() * 1e-3

        vf_loss.backward()
        pol_loss.backward()

        torch.nn.utils.clip_grad_norm_(curr_agent.critic.parameters(), 0.5)
        torch.nn.utils.clip_grad_norm_(curr_agent.policy.parameters(), 0.5)
        # curr_agent.policy_optimizer.step()
        curr_agent.optimizer.step()

        return vf_loss, pol_loss

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

    def save(self, filename):
        """
        Save trained parameters of all agents into one file
        """
        self.prep_training(device='cpu')  # move parameters to CPU before saving
        save_dict = {
            'n_agents': self.n_agents,
            'input_dim': self.input_dim,
            'act_dim': self.act_dim,
            'gamma': self.gamma,
            'tau': self.tau,
            'hidden_dim': self.hidden_dim,
            'shared_params': self.shared_params,
            'discrete_action': self.discrete_action,
            'init_explo_rate': self.init_explo_rate,
            'explo_strat': self.explo_strat,
            'agent_params': [a.get_params() for a in self.agents]}
        torch.save(save_dict, filename)

    def load_cp(self, cp_path):
        save_dict = torch.load(cp_path, map_location=torch.device('cpu'))
        for a, params in zip(self.agents, save_dict['agent_params']):
            a.load_params(params)
