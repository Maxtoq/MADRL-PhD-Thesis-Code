import torch
import numpy as np
from torch import nn

from .lm import GRUEncoder, GRUDecoder
from src.algo.nn_modules.mlp import MLPNetwork
    

class LanguageLearner(nn.Module):

    """ 
    Class to manage and train the language modules: the Language Encoder, the 
    Observation Encoder and the Decoder. 
    """

    def __init__(self, args, word_encoder, obs_dim, context_dim, n_agents, 
                 device="cpu"):
        super(LanguageLearner, self).__init__()
        self.train_device = device

        self.device = self.train_device

        self.word_encoder = word_encoder

        self.lang_encoder = GRUEncoder(
            context_dim, 
            args.lang_hidden_dim, 
            args.lang_embed_dim, 
            self.word_encoder,
            device=device)

        self.obs_encoder = MLPNetwork(
            args.hidden_dim, args.context_dim, args.hidden_dim)

        self.decoder = GRUDecoder(
            context_dim, 
            args.lang_embed_dim, 
            self.word_encoder, 
            embed_layer=self.lang_encoder.embed_layer,
            device=device)

        self.clip_loss = nn.CrossEntropyLoss()
        self.captioning_loss = nn.NLLLoss()

        self.optim = torch.optim.Adam( 
            self.parameters(),
            # list(self.lang_encoder.parameters()) +
            # list(self.obs_encoder.parameters()) +
            # list(self.decoder.parameters()),
            lr=args.lang_lr)

    def prep_rollout(self, device=None):
        self.device = self.train_device if device is None else device
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
        :param sentence_batch (list(list(int))): Batch of enoded sentences.

        :return context_batch (torch.Tensor): Batch of context vectors, 
            dim=(batch_size, context_dim).
        """
        context_batch = self.lang_encoder(sentence_batch).squeeze(0)
        return context_batch

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

    # def get_save_dict(self):
    #     save_dict = {
    #         "lang_encoder": self.lang_encoder.state_dict(),
    #         "decoder": self.decoder.state_dict()}
    #     return save_dict

    # def load_params(self, save_dict):
    #     self.lang_encoder.load_state_dict(save_dict["lang_encoder"])
    #     self.decoder.load_state_dict(save_dict["decoder"])