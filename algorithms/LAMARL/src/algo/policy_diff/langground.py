import torch
from torch import nn

from src.algo.language.lm import GRUEncoder, OneHotEncoder
from src.algo.nn_modules.mlp import MLPNetwork
    

class LanguageGrounder(nn.Module):

    """ 
    Class to manage and train the language grounding tools: a Language Encoder and 
    Observation Encoder.
    """

    def __init__(self, input_dim, context_dim, hidden_dim, embed_dim, policy_layer_N, 
                 lr, vocab, max_message_len, device="cuda:0"):
        super(LanguageGrounder, self).__init__()
        self.device = device

        self.word_encoder = OneHotEncoder(vocab, max_message_len)

        self.lang_encoder = GRUEncoder(
            context_dim, 
            hidden_dim, 
            embed_dim, 
            self.word_encoder,
            device=device)

        self.obs_encoder = MLPNetwork(
            input_dim, context_dim, hidden_dim, policy_layer_N)

        self.clip_loss = nn.CrossEntropyLoss()

        self.optim = torch.optim.Adam( 
            self.parameters(),
            lr=lr)

    def prep_rollout(self, device):
        self.device = device
        self.eval()
        self.to(self.device)
        self.lang_encoder.device = self.device

    def prep_training(self, device):
        self.device = device
        self.train()
        self.to(self.device)
        self.lang_encoder.device = self.device

    def encode_sentences(self, sentence_batch):
        """ 
        Encode a batch of sentences. 
        :param sentence_batch (list(list(int))): Batch of enoded sentences.

        :return context_batch (torch.Tensor): Batch of context vectors, 
            dim=(batch_size, context_dim).
        """
        context_batch = self.lang_encoder(sentence_batch).squeeze(0)
        return context_batch

    def get_save_dict(self):
        save_dict = {
            "lang_encoder": self.lang_encoder.state_dict(),
            "obs_encoder": self.obs_encoder.state_dict()}
        return save_dict

    def load_params(self, save_dict):
        self.lang_encoder.load_state_dict(save_dict["lang_encoder"])
        self.obs_encoder.load_state_dict(save_dict["obs_encoder"])