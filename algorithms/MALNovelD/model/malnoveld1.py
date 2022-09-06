import torch
import numpy as np

from torch import nn

from .modules.lnoveld import LNovelD
from .modules.lm import OneHotEncoder, GRUEncoder, GRUDecoder
from .modules.obs import ObservationEncoder
from .modules.comm import CommunicationPolicy
from .modules.maddpg1 import MADDPG


class MALNovelD:
    """
    Class for training Multi-Agent Language-NovelD, generating actions and 
    executing training for all agents.
    """
    def __init__(self, 
            obs_dim, 
            act_dim, 
            embed_dim,
            n_agents,
            vocab, 
            lr,
            gamma=0.99,
            tau=0.01,
            temp=1.0,
            hidden_dim=64, 
            context_dim=16,
            init_explo_rate=1.0,
            noveld_lr=1e-4,
            noveld_scale=0.5,
            noveld_trade_off=1,
            discrete_action=False,
            shared_params=False,
            pol_algo="maddpg"):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.embed_dim = embed_dim
        self.n_agents = n_agents
        self.vocab = vocab
        self.lr = lr
        self.gamma = gamma
        self.tau = tau
        self.temp = temp
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim
        self.init_explo_rate = init_explo_rate
        self.noveld_lr = noveld_lr
        self.noveld_scale = noveld_scale
        self.noveld_trade_off = noveld_trade_off
        self.discrete_action = discrete_action
        self.shared_params = shared_params
        self.pol_algo = pol_algo
        
        # Modules
        self.obs_encoder = ObservationEncoder(
            obs_dim, context_dim, hidden_dim, n_hidden_layers=0)
        self.word_encoder = OneHotEncoder(vocab)
        self.sentence_encoder = GRUEncoder(context_dim, self.word_encoder)
        self.decoder = GRUDecoder(context_dim, self.word_encoder)
        self.comm_policy = CommunicationPolicy(context_dim, hidden_dim)
        self.lnoveld = LNovelD(
            n_agents * obs_dim, n_agents * context_dim, 
            embed_dim, hidden_dim, noveld_lr,
            noveld_scale, noveld_trade_off)

        # Policy module
        if pol_algo == "maddpg":
            self.policy = MADDPG(
                n_agents, context_dim, act_dim, self.obs_encoder,
                lr, gamma, tau, hidden_dim,
                # n_agents, obs_dim, act_dim, lr, gamma, tau, hidden_dim,
                discrete_action, shared_params, init_explo_rate)
        else:
            print("Wrong algorithm name for the policy, must be in [maddpg]")
            exit()

        # Language losses
        self.clip_loss = nn.CrossEntropyLoss()
        self.captioning_loss = nn.NLLLoss()
        # Optimizer for learning language
        self.language_optimizer = torch.optim.Adam(
            list(self.obs_encoder.parameters()) + 
            list(self.sentence_encoder.parameters()), 
            lr=lr)

        self.device = 'cpu'

    def update_exploration_rate(self, epsilon):
        self.policy.scale_noise(epsilon)

    def update_all_targets(self):
        self.policy.update_all_targets()

    def reset(self):
        self.lnoveld.reset()
        self.policy.reset_noise()

    def prep_training(self, device='cpu'):
        if type(device) is str:
            device = torch.device(device)
        self.policy.prep_training(device)
        self.obs_encoder.train()
        self.obs_encoder = self.obs_encoder.to(device)
        self.sentence_encoder.train()
        self.sentence_encoder = self.sentence_encoder.to(device)
        self.sentence_encoder.device = device
        self.decoder.train()
        self.decoder = self.decoder.to(device)
        self.decoder.device = device
        self.comm_policy.train()
        self.comm_policy = self.comm_policy.to(device)
        self.lnoveld.set_train(device)
        self.device = device

    def prep_rollouts(self, device='cpu'):
        if type(device) is str:
            device = torch.device(device)
        self.policy.prep_rollouts(device)
        self.obs_encoder.eval()
        self.obs_encoder = self.obs_encoder.to(device)
        self.sentence_encoder.eval()
        self.sentence_encoder = self.sentence_encoder.to(device)
        self.sentence_encoder.device = device
        self.decoder.eval()
        self.decoder = self.decoder.to(device)
        self.decoder.device = device
        self.comm_policy.eval()
        self.comm_policy = self.comm_policy.to(device)
        self.lnoveld.set_eval(device)
        self.device = device

    def save(self, filename):
        """
        Save trained parameters of all agents into one file
        """
        self.prep_training(device='cpu')  # move parameters to CPU before saving
        save_dict = {
            'obs_dim': self.obs_dim,
            'act_dim': self.act_dim,
            'embed_dim': self.embed_dim,
            'n_agents': self.n_agents,
            'vocab': self.vocab,
            'lr': self.lr,
            'gamma': self.gamma,
            'tau': self.tau,
            'temp': self.temp,
            'hidden_dim': self.hidden_dim,
            'context_dim': self.context_dim,
            'init_explo_rate': self.init_explo_rate,
            'noveld_lr': self.noveld_lr,
            'noveld_scale': self.noveld_scale,
            'noveld_trade_off': self.noveld_trade_off,
            'discrete_action': self.discrete_action,
            'shared_params': self.shared_params,
            'pol_algo': self.pol_algo,
            'obs_encoder_params': self.obs_encoder.get_params(),
            'word_encoder': self.word_encoder,
            'sentence_encoder_params': self.sentence_encoder.get_params(),
            'decoder_params': self.decoder.get_params(),
            'comm_policy_params': self.comm_policy.get_params(),
            'obs_noveld_params': self.lnoveld.obs_noveld.get_params(),
            'lang_noveld_params': self.lnoveld.lang_noveld.get_params(),
            'agent_params': [a.get_params() for a in self.policy.agents]}
        torch.save(save_dict, filename)

    def get_intrinsic_rewards(self, observations, descriptions):
        """
        Get intrinsic reward of the multi-agent system.
        Inputs:
            observations (list(numpy.ndarray)): List of observations, one for 
                each agent.
            descriptions (list(list(str))): List of sentences describing 
                observations, one for each agent.
        Outputs:
            int_rewards (list): List of agents' intrinsic rewards.
        """
        # Concatenate observations
        cat_obs = torch.Tensor(np.concatenate(observations)).unsqueeze(0).to(
            self.device)

        # Encode descriptions
        encoded_descr = self.sentence_encoder(descriptions).detach()
        cat_descr = encoded_descr.view(1, -1)

        # Get reward
        int_reward = self.lnoveld.get_reward(cat_obs, cat_descr)
        return [int_reward] * self.n_agents

    def step(self, observations, descriptions=None, explore=False):
        """
        Perform a step, calling all modules.
        Inputs:
            observations (list(numpy.ndarray)): List of observations, one for 
                each agent.
            descriptions (list(list(str))): List of sentences describing 
                observations, one for each agent, if None evaluation step.
            explore (bool): Whether or not to perform exploration.
        Outputs:
            actions (list(torch.Tensor)): List of actions, one tensor of dim 
                (1, action_dim) for each agent.
        """
        # Encode observations
        torch_obs = torch.Tensor(np.array(observations)).to(self.device)

        # If the LNovelD network is empty (first step of new episode)
        if descriptions is not None and self.lnoveld.is_empty():
            # Encode descriptions
            encoded_descr = self.sentence_encoder(descriptions).to(
                self.device).detach()
            # Send the concatenated contexts and sentences encodings to lnoveld
            self.lnoveld.get_reward(
                torch_obs.view(1, -1),
                encoded_descr.view(1, -1)
            )

        # Get actions
        actions = self.policy.step(torch_obs, explore)
        # actions = self.policy.step(torch_obs, explore)
        return actions

    def update_policy(self, agents_batch):
        vf_losses = []
        pol_losses = []
        for a_i in range(self.n_agents):
            vf_loss, pol_loss = self.policy.update(
                agents_batch[a_i], a_i)
            vf_losses.append(vf_loss.item())
            pol_losses.append(pol_loss.item())

        return vf_losses, pol_losses

    def update_lnoveld(self):
        return self.lnoveld.train()

    def update_language_modules(self, language_batch):
        self.language_optimizer.zero_grad()
        obs_batch, sent_batch = language_batch

        # Encode observations
        obs_tensor = torch.Tensor(np.array(obs_batch)).to(self.device)
        context_batch = self.obs_encoder(obs_tensor)
        # Encode sentences
        lang_context_batch = self.sentence_encoder(sent_batch)
        lang_context_batch = lang_context_batch.squeeze()

        # CLIP loss
        # Compute similarity
        norm_context_batch = context_batch / context_batch.norm(
            dim=1, keepdim=True)
        lang_context_batch = lang_context_batch / lang_context_batch.norm(
            dim=1, keepdim=True)
        sim = norm_context_batch @ lang_context_batch.t() * self.temp
        # mean_sim = sim.diag().mean()
        # Compute loss
        labels = torch.arange(len(obs_batch)).to(self.device)
        loss_o = self.clip_loss(sim, labels)
        loss_l = self.clip_loss(sim.t(), labels)
        clip_loss = (loss_o + loss_l) / 2

        # Step
        clip_loss.backward()
        self.language_optimizer.step()

        return float(clip_loss)

    def update(self, agents_batch, language_batch):
        """
        Perform a learning step on all modules of the model.
        Inputs:
            agents_batch (list(tuple(torch.Tensor))): List of (observation, 
                action, reward, next observation, and episode end bool) tuples
                (one for each agent).
            language_batch (tuple(list)): Tuple containing a batch of 
                observations (list of numpy arrays) and a batch of corresponding
                language descriptions (list of lists of words).
        Outputs:
            vf_losses (list(float)): Value losses (one for each agent).
            pol_losses (list(float)): Policy losses (one for each agent).
            clip_loss (float): Loss of observation and description learning.
            lnd_obs_loss (float): Loss of the observation prediction in the
                L-NovelD network.
            lnd_lang_loss (float): Loss of the language prediction in the
                L-NovelD network.
        """
        # POLICY LEARNING
        vf_losses = []
        pol_losses = []
        for a_i in range(self.n_agents):
            # self.language_optimizer.zero_grad()
            obs, acs, rews, next_obs, dones = agents_batch[a_i]
            # with torch.no_grad():
            #     self.obs_encoder.eval()
            #     enc_obs = [self.obs_encoder(o) for o in obs]
            #     enc_next_obs = [self.obs_encoder(n_o) for n_o in next_obs]
            #     self.obs_encoder.train()
            # agent_batch = (enc_obs, acs, rews, enc_next_obs, dones)
            agent_batch = (obs, acs, rews, next_obs, dones)
            vf_loss, pol_loss = self.policy.update(agent_batch, a_i)
            # self.language_optimizer.step()
            vf_losses.append(vf_loss.item())
            pol_losses.append(pol_loss.item())

        # LNOVELD LEARNING
        lnd_obs_loss, lnd_lang_loss = self.lnoveld.train()
            
        # LANGUAGE LEARNING
        self.language_optimizer.zero_grad()
        obs_batch, sent_batch = language_batch

        # Encode observations
        obs_tensor = torch.Tensor(np.array(obs_batch)).to(self.device)
        context_batch = self.obs_encoder(obs_tensor)
        # Encode sentences
        lang_context_batch = self.sentence_encoder(sent_batch)
        lang_context_batch = lang_context_batch.squeeze()

        # CLIP loss
        # Compute similarity
        norm_context_batch = context_batch / context_batch.norm(
            dim=1, keepdim=True)
        lang_context_batch = lang_context_batch / lang_context_batch.norm(
            dim=1, keepdim=True)
        sim = norm_context_batch @ lang_context_batch.t() * self.temp
        # mean_sim = sim.diag().mean()
        # Compute loss
        labels = torch.arange(len(obs_batch)).to(self.device)
        loss_o = self.clip_loss(sim, labels)
        loss_l = self.clip_loss(sim.t(), labels)
        clip_loss = (loss_o + loss_l) / 2

        # Step
        clip_loss.backward()
        self.language_optimizer.step()

        return vf_losses, pol_losses, float(clip_loss), lnd_obs_loss, lnd_lang_loss