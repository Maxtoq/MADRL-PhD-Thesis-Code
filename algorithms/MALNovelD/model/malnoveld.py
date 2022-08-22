import torch
import numpy as np

from torch import nn

from modules.lnoveld import LNovelD
from modules.lm import OneHotEncoder, GRUEncoder, GRUDecoder
from modules.obs import ObservationEncoder
from modules.comm import CommunicationPolicy
from modules.maddpg import MADDPG


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
            init_explo_rate=1.0,
            context_dim=64,
            noveld_lr=1e-4,
            noveld_scale=0.5,
            noveld_trade_off=1,
            discrete_action=False,
            shared_params=False,
            pol_algo="maddpg"):
        self.n_agents =n_agents
        self.temp = temp
        
        # Modules
        self.obs_encoder = ObservationEncoder(obs_dim, context_dim, hidden_dim)
        self.word_encoder = OneHotEncoder(vocab)
        self.sentence_encoder = GRUEncoder(context_dim, self.word_encoder)
        # self.decoder = GRUDecoder(context_dim, self.word_encoder)
        # self.comm_policy = CommunicationPolicy(context_dim)
        self.lnoveld = LNovelD(
            n_agents * obs_dim, n_agents * context_dim, 
            embed_dim, hidden_dim, noveld_lr,
            noveld_scale, noveld_trade_off)

        # Policy module
        if pol_algo == "maddpg":
            self.policy = MADDPG(
                n_agents, context_dim, act_dim, lr, gamma, tau, hidden_dim,
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

    def update_policy_exploration(self, epsilon):
        self.policy.scale_noise(epsilon)

    def step(self, observations, descriptions, explore=False):
        """
        Perform a step, calling all modules.
        Inputs:
            observations (list(numpy.ndarray)): List of observations, one for 
                each agent.
            descriptions (list(list(str))): List of sentences describing 
                observations, one for each agent.
            explore (bool): Whether or not to perform exploration.
        """
        # Encode observations
        torch_obs = torch.Tensor(np.array(observations))
        internal_contexts = self.obs_encoder(torch_obs)

        # If the LNovelD netword is empty (first step of new episode)
        if self.lnoveld.is_empty():
            # Encode descriptions
            encoded_sentences = self.sentence_encoder(descriptions)
            # Send the concatenated contexts and sentences encodings to lnoveld
            self.lnoveld.get_reward(
                internal_contexts.view(-1, 1),
                encoded_sentences.view(-1, 1)
            )

        # Get actions
        actions = self.policy.step(internal_contexts, explore)
        return actions

    def update(self, agents_batch, language_batch):
        # Agents update
        for a_i in range(self.n_agents):
            self.maddpg.update(agents_batch[a_i], a_i)
            
        # Language learning
        self.language_optimizer.zero_grad()
        obs_batch, sent_batch = language_batch

        # Encode observations
        obs_tensor = torch.Tensor(obs_batch)
        context_batch = self.obs_encoder(obs_tensor)
        # Encode sentences
        lang_context_batch = self.lang_encoder(lang_context_batch)
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
        labels = torch.arange(len(obs_batch))
        loss_o = self.clip_loss(sim, labels)
        loss_l = self.clip_loss(sim.t(), labels)
        clip_loss = (loss_o + loss_l) / 2

        # Step
        clip_loss.backward()
        self.language_optimizer.step()