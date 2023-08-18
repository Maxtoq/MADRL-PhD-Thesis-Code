import copy
import torch
from torch import nn

from .modules.lm import OneHotEncoder, GRUEncoder, GRUDecoder
from .modules.obs import ObservationEncoder
from .modules.lang_buffer import LanguageBuffer
from .modules.networks import MLPNetwork
from .policy.mappo import MAPPO


class LanguageLearner:

    def __init__(self, args, obs_dim, context_dim, hidden_dim, vocab):
        self.lr = args.lang_lr
        self.n_train_iter = args.lang_n_train_iter
        self.batch_size = args.lang_batch_size
        self.temp = args.lang_temp
        self.clip_weight = args.lang_clip_weight
        self.capt_weight = args.lang_capt_weight
        self.obs_learn_capt = args.lang_obs_learn_capt

        self.word_encoder = OneHotEncoder(vocab)

        self.obs_encoder = ObservationEncoder(obs_dim, context_dim, hidden_dim)
        self.lang_encoder = GRUEncoder(
            context_dim, hidden_dim, self.word_encoder)
        self.decoder = GRUDecoder(context_dim, self.word_encoder)

        self.clip_loss = nn.CrossEntropyLoss()
        self.captioning_loss = nn.NLLLoss()

        self.optim = torch.optim.Adam(
            list(self.obs_encoder.parameters()) + 
            list(self.lang_encoder.parameters()) + 
            list(self.decoder.parameters()), 
            lr=lr)

        self.buffer = LanguageBuffer(args.lang_buffer_size)

    def encode(self, sentence_batch):
        """ 
        Encode a batch of sentences. 
        :param sentence_batch (list(list(str))): Batch of sentences.

        :return context_batch (torch.Tensor): Batch of context vectors, 
            dim=(batch_size, context_dim).
        """
        context_batch = self.lang_encoder(sentence_batch)
        return context_batch

    def generate_sentences(self, context_batch):
        """ 
        Generate sentences from a batch of context vectors. 
        :param context_batch (torch.Tensor): Batch of context vectors,
            dim=(batch_size, context_dim).
        
        :return gen_sent_batch (list(list(str))): Batch of generated sentences.
        """
        _, preds = self.decoder(context_batch)

        gen_sent_batch = self.word_encoder.decode(preds)
        return gen_sent_batch

    def compute_losses(self, obs_batch, sent_batch):
        # Encode observations
        obs_tensor = torch.Tensor(np.array(obs_batch))
        obs_context_batch = self.obs_encoder(obs_tensor)

        # Encode sentences
        lang_context_batch = self.lang_encoder(sent_batch)
        lang_context_batch = lang_context_batch.squeeze()

        # Compute similarity
        norm_context_batch = obs_context_batch / obs_context_batch.norm(
            dim=1, keepdim=True)
        lang_context_batch = lang_context_batch / lang_context_batch.norm(
            dim=1, keepdim=True)
        sim = norm_context_batch @ lang_context_batch.t() * self.temp
        mean_sim = sim.diag().mean()

        # Compute CLIP loss
        labels = torch.arange(len(obs_batch))
        loss_o = self.clip_loss(sim, labels)
        loss_l = self.clip_loss(sim.t(), labels)
        clip_loss = (loss_o + loss_l) / 2
        
        # Decoding
        encoded_targets = word_encoder.encode_batch(sent_batch)
        if not self.obs_learn_capt:
            obs_context_batch = obs_context_batch.detach()
        decoder_outputs, _ = dec(obs_context_batch, encoded_targets)

        # Compute Captioning loss
        dec_loss = 0
        for d_o, e_t in zip(decoder_outputs, encoded_targets):
            e_t = torch.argmax(e_t, dim=1)
            dec_loss += self.captioning_loss(d_o, e_t)
        
        return clip_loss, dec_loss, mean_sim

    def train(self):
        clip_losses = []
        dec_losses = []
        for it in range(self.n_train_iter):
            self.optim.zero_grad()
            # Sample batch from buffer
            obs_batch, sent_batch = self.buffer.sample(self.batch_size)

            # Compute losses
            clip_loss, dec_loss, _ = self.compute_losses(obs_batch, sent_batch)

            # Update
            tot_loss = self.clip_weight * clip_loss + self.capt_weight * dec_loss
            tot_loss.backward()
            self.optim.step()

            clip_losses.append(clip_loss.item())
            dec_losses.append(dec_loss.item() / batch_size)
        
        clip_loss = sum(clip_losses) / len(clip_losses)
        dec_loss = sum(dec_losses) / len(dec_loss)
        
        return clip_loss, dec_loss


class CommunicationPolicy:

    def __init__(self, context_dim, hidden_dim, obs_encoder, lang_encoder, decoder):
        # Pretrained modules
        self.obs_encoder = obs_encoder
        self.lang_encoder = lang_encoder
        self.decoder = decoder
        # Policy
        self.context_encoder = MLPNetwork(
            context_dim * 2, 
            context_dim, 
            hidden_dim,
            n_hidden_layers=0)
        self.message_gen = copy.deepcopy(self.decoder)




class LMC:
    """
    Language-Memory for Communication using a pre-defined discrete language.
    """
    def __init__(self, args, n_agents, 
                 obs_dim, shared_obs_dim, act_dim, 
                 vocab, device):
        self.args = args
        self.n_agents = n_agents

        # Modules
        self.language_learner = LanguageLearner(
            obs_dim, args.context_dim, args.hidden_dim, vocab)
        self.comm_policy = CommunicationPolicy(context_dim, hidden_dim)
        if self.policy_algo == "mappo":
            self.policy = MAPPO(
                args, n_agents, obs_space, shared_obs_space, args.context_dim,
                act_space, device)

        self.device = device

    def prep_training(self, device=None):
        pass

    def prep_rollout(self, device=None):
        pass

    def start_episode(self, obs):
        pass

    def comm_n_act(self, obs):
        pass

    def train(self):
        pass

    def save(self):
        pass
