from torch import nn

from modules.lm import OneHotEncoder, GRUEncoder, GRUDecoder
from modules.obs import ObservationEncoder
from modules.lang_buffer import LanguageBuffer
from policy.mappo import MAPPO


# TODO Add arguments for language parameters: lang_lr, lang_buffer_size


class LanguageLearner:

    def __init__(self, obs_dim, context_dim, hidden_dim, vocab, lr):
        self.word_encoder = OneHotEncoder(vocab)

        self.obs_encoder = ObservationEncoder(
            obs_dim, args.context_dim, args.hidden_dim)
        self.lang_encoder = GRUEncoder(context_dim, self.word_encoder)
        self.decoder = GRUDecoder(context_dim, self.word_encoder)

        self.clip_loss = nn.CrossEntropyLoss()
        self.captioning_loss = nn.NLLLoss()

        self.opt = optim.Adam(
            list(self.obs_encoder.parameters()) + 
            list(self.lang_encoder.parameters()) + 
            list(self.decoder.parameters()), 
            lr=lr)

        self.buffer = LanguageBuffer(args.lang_buffer_size)

    def encode(self, sentence_batch):
        """ Encode a batch of sentences. """
        pass

    def generate_sentence(self, context_batch):
        """ Generate sentences from a batch of context vectors. """
        pass

    def train(self):
        pass


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
            obs_dim, args.context_dim, args.hidden_dim, vocab, args.lang_lr)
        # self.comm_policy = CommunicationPolicy(context_dim, hidden_dim)
        if self.policy_algo == "mappo":
            self.policy = MAPPO(
                args, n_agents, obs_space, shared_obs_space, args.context_dim,
                act_space, device)

        self.device = device
