import torch
import numpy as np
from torch import nn

from .lm import OneHotEncoder, GRUEncoder, GRUDecoder
from .obs import ObservationEncoder
from .lang_buffer import LanguageBuffer
    

class LanguageLearner:

    """ 
    Class to manage and train the language modules: the Language Encoder, the 
    Observation Encoder and the Decoder. 
    """

    def __init__(self, args, obs_dim, context_dim, parser, device="cpu"):
        self.train_device = device

        self.device = self.train_device

        self.word_encoder = OneHotEncoder(parser.vocab)

        # self.obs_encoder = ObservationEncoder(
        #     obs_dim, context_dim, args.lang_hidden_dim)
        self.lang_encoder = GRUEncoder(
            context_dim, 
            args.lang_hidden_dim, 
            args.lang_embed_dim, 
            self.word_encoder)
        self.decoder = GRUDecoder(
            context_dim, self.word_encoder, max_length=parser.max_message_len)

        self.clip_loss = nn.CrossEntropyLoss()
        self.captioning_loss = nn.NLLLoss()

        # self.clip_optim = torch.optim.Adam(
        #     list(self.obs_encoder.parameters()) + 
        #     list(self.lang_encoder.parameters()),
        #     lr=args.lang_clip_lr)

    def prep_rollout(self, device=None):
        self.device = self.train_device if device is None else device
        # self.obs_encoder.eval()
        # self.obs_encoder.to(self.device)
        # self.obs_encoder.device = self.device
        self.lang_encoder.eval()
        self.lang_encoder.to(self.device)
        self.lang_encoder.device = self.device
        self.decoder.eval()
        self.decoder.to(self.device)
        self.decoder.device = self.device

    def prep_training(self, device=None):
        if device is not None:
            self.train_device = device
        self.device = self.train_device
        # self.obs_encoder.train()
        # self.obs_encoder.to(self.device)
        # self.obs_encoder.device = self.device
        self.lang_encoder.train()
        self.lang_encoder.to(self.device)
        self.lang_encoder.device = self.device
        self.decoder.train()
        self.decoder.to(self.device)
        self.decoder.device = self.device
    
    def store(self, obs, sent):
        self.buffer.store(obs, sent)

    def encode_sentences(self, sentence_batch):
        """ 
        Encode a batch of sentences. 
        :param sentence_batch (list(list(str))): Batch of sentences.

        :return context_batch (torch.Tensor): Batch of context vectors, 
            dim=(batch_size, context_dim).
        """
        context_batch = self.lang_encoder(sentence_batch).squeeze(0)
        return context_batch
    
    # def encode_observations(self, obs_batch):
    #     """
    #     :param obs_batch: (torch.Tensor) Batch of observation, 
    #         dim=(batch_size, obs_dim).
    #     """
    #     context_batch = self.obs_encoder(obs_batch)
    #     return context_batch

    def generate_sentences(self, context_batch):
        """ 
        Generate sentences from a batch of context vectors. 
        :param context_batch (np.ndarray): Batch of context vectors,
            dim=(batch_size, context_dim).
        
        :return gen_sent_batch (list(list(str))): Batch of generated sentences.
        """
        context_batch = torch.from_numpy(context_batch).to(self.device)
        _, sentences = self.decoder(context_batch)
        return sentences

    # def compute_losses(self, obs_batch, sent_batch):
    #     # Encode observations
    #     obs_tensor = torch.from_numpy(np.array(obs_batch, dtype=np.float32))
    #     obs_context_batch = self.obs_encoder(obs_tensor)

    #     # Encode sentences
    #     lang_context_batch = self.lang_encoder(sent_batch)
    #     lang_context_batch = lang_context_batch.squeeze()

    #     # Compute similarity
    #     norm_context_batch = obs_context_batch / obs_context_batch.norm(
    #         dim=1, keepdim=True)
    #     lang_context_batch = lang_context_batch / lang_context_batch.norm(
    #         dim=1, keepdim=True)
    #     sim = norm_context_batch @ lang_context_batch.t() * self.temp
    #     mean_sim = sim.diag().mean()

    #     # Compute CLIP loss
    #     labels = torch.arange(len(obs_batch)).to(self.train_device)
    #     loss_o = self.clip_loss(sim, labels)
    #     loss_l = self.clip_loss(sim.t(), labels)
    #     clip_loss = (loss_o + loss_l) / 2
        
    #     # Decoding
    #     encoded_targets = self.word_encoder.encode_batch(sent_batch)
    #     if not self.obs_learn_capt:
    #         obs_context_batch = obs_context_batch.detach()
    #     decoder_outputs, _ = self.decoder(obs_context_batch, encoded_targets)

    #     # Compute Captioning loss
    #     dec_loss = 0
    #     for d_o, e_t in zip(decoder_outputs, encoded_targets):
    #         e_t = torch.argmax(e_t, dim=1).to(self.train_device)
    #         dec_loss += self.captioning_loss(d_o[:e_t.size(0)], e_t)
        
    #     return clip_loss, dec_loss, mean_sim

    def get_save_dict(self):
        save_dict = {
            # "obs_encoder": self.obs_encoder.state_dict(),
            "lang_encoder": self.lang_encoder.state_dict(),
            "decoder": self.decoder.state_dict()}
        return save_dict

    def load_params(self, save_dict):
        # self.obs_encoder.load_state_dict(save_dict["obs_encoder"])
        self.lang_encoder.load_state_dict(save_dict["lang_encoder"])
        self.decoder.load_state_dict(save_dict["decoder"])