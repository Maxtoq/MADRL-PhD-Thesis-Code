import torch
import numpy as np

from .qmix import QMIX, QMIXAgent
from .lnoveld import NovelD, LNovelD
from .lm import OneHotEncoder


class QMIXAgent_NovelD(QMIXAgent):

    def __init__(self, obs_dim, act_dim, 
                 hidden_dim=64, init_explo=1.0, device="cpu",
                 embed_dim=16, nd_lr=1e-4, nd_scale_fac=0.5, nd_hidden_dim=64):
        super(QMIXAgent_NovelD, self).__init__(
            obs_dim, act_dim, hidden_dim, init_explo, device)
        self.noveld = NovelD(
            obs_dim, embed_dim, nd_hidden_dim, nd_lr, nd_scale_fac)

    def get_actions(self, obs, last_acts, qnet_rnn_states, explore=False):
        # If we are starting a new episode, compute novelty for first observation
        if self.noveld.is_empty():
            self.noveld.get_reward(obs)

        return super().get_actions(obs, last_acts, qnet_rnn_states, explore)

    def get_intrinsic_reward(self, next_obs):
        intr_reward = self.noveld.get_reward(next_obs)
        return intr_reward
    
    def train_noveld(self):
        return self.noveld.train_predictor()

    def reset_noveld(self):
        self.noveld.init_new_episode()
        
    def get_params(self):
        return {'q_net': self.q_net.state_dict(),
                'target_q_net': self.target_q_net.state_dict(),
                'noveld': self.noveld.get_params()}

    def load_params(self, params):
        self.q_net.load_state_dict(params['q_net'])
        self.target_q_net.load_state_dict(params['target_q_net'])
        self.noveld.load_params(params['noveld'])


class QMIX_PANovelD(QMIX):
    """ 
    Class impelementing QMIX with Per Agent NovelD (QMIX_PANovelD), meaning
    that each agent has its own local NovelD model to compute a local 
    intrinsic reward.
    """
    def __init__(self, nb_agents, obs_dim, act_dim, lr, 
                 gamma=0.99, tau=0.01, hidden_dim=64, shared_params=False, 
                 init_explo_rate=1.0, max_grad_norm=None, device="cpu",
                 embed_dim=16, nd_lr=1e-4, nd_scale_fac=0.5, nd_hidden_dim=64):
        super(QMIX_PANovelD, self).__init__(
            nb_agents, obs_dim, act_dim, lr, gamma, tau, hidden_dim, 
            shared_params, init_explo_rate, max_grad_norm, device)
        
        # Create agent policies
        if not shared_params:
            self.agents = [QMIXAgent_NovelD(
                    obs_dim, act_dim, hidden_dim, init_explo_rate,
                    device, embed_dim, nd_lr, nd_scale_fac, nd_hidden_dim)
                for _ in range(nb_agents)]
        else:
            self.agents = [QMIXAgent_NovelD(
                    obs_dim, act_dim, hidden_dim, init_explo_rate,
                    device, embed_dim, nd_lr, nd_scale_fac, nd_hidden_dim)]

        # Initiate optimiser with all parameters
        self.parameters = []
        for ag in self.agents:
            self.parameters += ag.q_net.parameters()
        self.parameters += self.mixer.parameters()
        self.optimizer = torch.optim.RMSprop(self.parameters, lr)
        
    def get_intrinsic_rewards(self, next_obs_list):
        """
        Get intrinsic rewards for all agents.
        Inputs:
            next_obs_list (list): List of agents' observations at next 
                step.
        Outputs:
            int_rewards (list): List of agents' intrinsic rewards.
        """
        int_rewards = []
        for a_i, next_obs in enumerate(next_obs_list):
            a_i = 0 if self.shared_params else a_i
            int_reward = self.agents[a_i].get_intrinsic_reward(
                torch.Tensor(next_obs).unsqueeze(0).to(self.device))
            int_rewards.append(int_reward)
        return int_rewards

    def train_on_batch(self, batch):
        """
        Update all agents and local NovelD models.
        Inputs:
            batch (tuple(torch.Tensor)): Tuple of batches of experiences for
                the agents to train on.
        Outputs:
            qtot_loss (float): QMIX loss.
            nd_loss (float): MA-NovelD loss.
        """
        qtot_loss = super().train_on_batch(batch)

        # NovelD updates
        nd_losses = []
        for agent in self.agents:
            nd_losses.append(agent.noveld.train_predictor())

        return qtot_loss, np.mean(nd_losses)

    def reset_noveld(self):
        for agent in self.agents:
            agent.reset_noveld()
    
    def prep_training(self, device='cpu'):
        super().prep_training(device)
        for agent in self.agents:
            agent.noveld.set_train(device)
    
    def prep_rollouts(self, device='cpu'):
        super().prep_rollouts(device)
        for agent in self.agents:
            agent.noveld.set_eval(device)

    def save(self, filename):
        self.prep_training(device='cpu')
        save_dict = {
            'nb_agents': self.nb_agents,
            'obs_dim': self.obs_dim,
            'act_dim': self.act_dim,
            'lr': self.lr,
            'gamma': self.gamma,
            'tau': self.tau,
            'hidden_dim': self.hidden_dim,
            'shared_params': self.shared_params,
            'max_grad_norm': self.max_grad_norm,
            'agent_params': [a.get_params() for a in self.agents],
            'mixer_params': self.mixer.state_dict(),
            'target_mixer_params': self.target_mixer.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        torch.save(save_dict, filename)

    @classmethod
    def init_from_save(cls, filename):
        """
        Instantiate instance of this class from file created by 'save' method
        """
        save_dict = torch.load(filename, map_location=torch.device('cpu'))
        agent_params = save_dict.pop("agent_params")
        mixer_params = save_dict.pop("mixer_params")
        target_mixer_params = save_dict.pop("target_mixer_params")
        optimizer = save_dict.pop("optimizer")
        instance = cls(**save_dict)
        for a, params in zip(instance.agents, agent_params):
            a.load_params(params)
        instance.mixer.load_state_dict(mixer_params)
        instance.target_mixer.load_state_dict(target_mixer_params)
        instance.optimizer.load_state_dict(optimizer)
        return instance


class QMIX_MANovelD(QMIX):
    """ 
    Class impelementing QMIX with Multi-Agent NovelD (QMIX_MANovelD), meaning
    that we use a single NovelD model to compute the intrinsic reward of the
    multi-agent system.
    """
    def __init__(self, nb_agents, obs_dim, act_dim, lr, 
                 gamma=0.99, tau=0.01, hidden_dim=64, shared_params=False, 
                 init_explo_rate=1.0, max_grad_norm=None, device="cpu",
                 embed_dim=16, nd_lr=1e-4, nd_scale_fac=0.5, nd_hidden_dim=64):
        super(QMIX_MANovelD, self).__init__(
            nb_agents, obs_dim, act_dim, lr, gamma, tau, hidden_dim, 
            shared_params, init_explo_rate, max_grad_norm, device)
        # Init NovelD model for the multi-agent system
        self.ma_noveld = NovelD(
            nb_agents * obs_dim, embed_dim, nd_hidden_dim, nd_lr, nd_scale_fac)

    def get_actions(self, 
            obs_list, last_actions, qnets_hidden_states, explore=False):
        # If we are starting a new episode, compute novelty for first obs
        if self.ma_noveld.is_empty():
            cat_obs = torch.Tensor(
                np.concatenate(obs_list)).unsqueeze(0).to(self.device)
            self.ma_noveld.get_reward(cat_obs.view(1, -1))
        return super().get_actions(
            obs_list, last_actions, qnets_hidden_states, explore)

    def get_intrinsic_rewards(self, next_obs_list):
        """
        Get intrinsic reward of the multi-agent system.
        Inputs:
            next_obs_list (list): List of agents' observations at next 
                step.
        Outputs:
            int_rewards (list): List of agents' intrinsic rewards.
        """
        # Concatenate observations
        cat_obs = torch.Tensor(
            np.concatenate(next_obs_list)).unsqueeze(0).to(self.device)
        # Get reward
        int_reward = self.ma_noveld.get_reward(cat_obs)
        int_rewards = [int_reward] * self.nb_agents
        return int_rewards
    
    def train_on_batch(self, batch):
        """
        Update all agents and NovelD model.
        Inputs:
            batch (tuple(torch.Tensor)): Tuple of batches of experiences for
                the agents to train on.
        Outputs:
            qtot_loss (float): QMIX loss.
            nd_loss (float): MA-NovelD loss.
        """
        qtot_loss = super().train_on_batch(batch)

        # NovelD update
        nd_loss = self.ma_noveld.train_predictor()

        return qtot_loss, nd_loss

    def reset_noveld(self):
        self.ma_noveld.init_new_episode()
    
    def prep_training(self, device='cpu'):
        super().prep_training(device)
        self.ma_noveld.set_train(device)
    
    def prep_rollouts(self, device='cpu'):
        super().prep_rollouts(device)
        self.ma_noveld.set_eval(device)

    def save(self, filename):
        self.prep_training(device='cpu')
        save_dict = {
            'nb_agents': self.nb_agents,
            'obs_dim': self.obs_dim,
            'act_dim': self.act_dim,
            'lr': self.lr,
            'gamma': self.gamma,
            'tau': self.tau,
            'hidden_dim': self.hidden_dim,
            'shared_params': self.shared_params,
            'max_grad_norm': self.max_grad_norm,
            'agent_params': [a.get_params() for a in self.agents],
            'mixer_params': self.mixer.state_dict(),
            'target_mixer_params': self.target_mixer.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'manoveld_params': self.ma_noveld.get_params()
        }
        torch.save(save_dict, filename)

    @classmethod
    def init_from_save(cls, filename):
        """
        Instantiate instance of this class from file created by 'save' method
        """
        save_dict = torch.load(filename, map_location=torch.device('cpu'))
        agent_params = save_dict.pop("agent_params")
        mixer_params = save_dict.pop("mixer_params")
        target_mixer_params = save_dict.pop("target_mixer_params")
        optimizer = save_dict.pop("optimizer")
        manoveld_params = save_dict.pop("manoveld_params")
        instance = cls(**save_dict)
        for a, params in zip(instance.agents, agent_params):
            a.load_params(params)
        instance.mixer.load_state_dict(mixer_params)
        instance.target_mixer.load_state_dict(target_mixer_params)
        instance.optimizer.load_state_dict(optimizer)
        instance.ma_noveld.load_params(manoveld_params)
        return instance


class QMIX_MALNovelD(QMIX):
    """ 
    Class impelementing QMIX with Multi-Agent L-NovelD (QMIX_MAaLNovelD),
    meaning that we use a single L-NovelD model to compute the intrinsic reward
    of the multi-agent system.
    """
    def __init__(self, nb_agents, obs_dim, act_dim, lr, vocab,
                 gamma=0.99, tau=0.01, hidden_dim=64, shared_params=False, 
                 init_explo_rate=1.0, max_grad_norm=None, device="cpu",
                 embed_dim=16, nd_lr=1e-4, nd_scale_fac=0.5, nd_hidden_dim=64,
                 lnd_trade_off=1.0):
        super(QMIX_MALNovelD, self).__init__(
            nb_agents, obs_dim, act_dim, lr, gamma, tau, hidden_dim, 
            shared_params, init_explo_rate, max_grad_norm, device)
        # Init word encoder
        word_encoder = OneHotEncoder(vocab)

        # Init L-NovelD model for the multi-agent system
        self.ma_lnoveld = LNovelD(
            nb_agents * obs_dim, 
            embed_dim, 
            word_encoder, 
            nd_hidden_dim, 
            nd_lr, 
            nd_scale_fac, 
            lnd_trade_off)

    def get_actions(self, 
            obs_list, last_actions, qnets_hidden_states, descr_list, 
            explore=False):
        # If we are starting a new episode, compute novelty for first obs
        if self.ma_lnoveld.is_empty():
            # Concatenate observations
            cat_obs = torch.Tensor(
                np.concatenate(obs_list)).unsqueeze(0).to(self.device)
            # Concatenate all descriptions
            cat_descr = [[word for sublist in descr_list for word in sublist]]
            self.ma_lnoveld.get_reward(cat_obs.view(1, -1), cat_descr)
        return super().get_actions(
            obs_list, last_actions, qnets_hidden_states, explore)

    def get_intrinsic_rewards(self, next_obs_list, next_descr_list):
        """
        Get intrinsic reward of the multi-agent system.
        Inputs:
            next_obs_list (list): List of agents' observations at next step.
            next_descr_list (list): List of agents' descriptions at next step.
        Outputs:
            int_rewards (list): List of agents' intrinsic rewards.
        """
        # Concatenate observations
        cat_obs = torch.Tensor(
            np.concatenate(next_obs_list)).unsqueeze(0).to(self.device)
        # Concatenate all descriptions
        cat_descr = [[word for sublist in next_descr_list for word in sublist]]
        # Get reward
        int_rew, obs_int_rew, lang_int_rew = self.ma_lnoveld.get_reward(
            cat_obs, cat_descr)
        int_rewards = [int_rew] * self.nb_agents
        obs_int_rewards = [obs_int_rew] * self.nb_agents
        lang_int_rewards = [lang_int_rew] * self.nb_agents
        return int_rewards, obs_int_rewards, lang_int_rewards
    
    def train_on_batch(self, batch):
        """
        Update all agents and L-NovelD model.
        Inputs:
            batch (tuple(torch.Tensor)): Tuple of batches of experiences for
                the agents to train on.
        Outputs:
            qtot_loss (float): QMIX loss.
            nd_loss (float): MA-L-NovelD loss.
        """
        qtot_loss = super().train_on_batch(batch)

        # L-NovelD update
        lnd_obs_loss, lnd_lang_loss = self.ma_lnoveld.train()

        return qtot_loss, lnd_obs_loss, lnd_lang_loss

    def reset_noveld(self):
        self.ma_lnoveld.reset()
    
    def prep_training(self, device='cpu'):
        super().prep_training(device)
        self.ma_lnoveld.set_train(device)
    
    def prep_rollouts(self, device='cpu'):
        super().prep_rollouts(device)
        self.ma_lnoveld.set_train(device)

    def save(self, filename):
        self.prep_training(device='cpu')
        save_dict = {
            'nb_agents': self.nb_agents,
            'obs_dim': self.obs_dim,
            'act_dim': self.act_dim,
            'lr': self.lr,
            'gamma': self.gamma,
            'tau': self.tau,
            'hidden_dim': self.hidden_dim,
            'shared_params': self.shared_params,
            'max_grad_norm': self.max_grad_norm,
            'agent_params': [a.get_params() for a in self.agents],
            'mixer_params': self.mixer.state_dict(),
            'target_mixer_params': self.target_mixer.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'manoveld_params': self.ma_lnoveld.get_params()
        }
        torch.save(save_dict, filename)

    @classmethod
    def init_from_save(cls, filename):
        """
        Instantiate instance of this class from file created by 'save' method
        """
        save_dict = torch.load(filename, map_location=torch.device('cpu'))
        agent_params = save_dict.pop("agent_params")
        mixer_params = save_dict.pop("mixer_params")
        target_mixer_params = save_dict.pop("target_mixer_params")
        optimizer = save_dict.pop("optimizer")
        manoveld_params = save_dict.pop("manoveld_params")
        instance = cls(**save_dict)
        for a, params in zip(instance.agents, agent_params):
            a.load_params(params)
        instance.mixer.load_state_dict(mixer_params)
        instance.target_mixer.load_state_dict(target_mixer_params)
        instance.optimizer.load_state_dict(optimizer)
        instance.ma_lnoveld.load_params(manoveld_params)
        return instance