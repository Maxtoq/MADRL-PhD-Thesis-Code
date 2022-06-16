import torch

from torch import nn

from modules.lnoveld import LNovelD


class ObservationEncoder(nn.Module):

    def __init__(self, obs_dim, embedding_dim, hidden_dim):
        super(ObservationEncoder, self).__init__()

class GRULanguageEncoder(nn.Module):

    def __init__(self, vocab, context_dim):
        super(GRULanguageEncoder, self).__init__()

class GRUDecoder(nn.Module):

    def __init__(self, vocab, context_dim):
        super(GRUDecoder, self).__init__()

class Policy(nn.Module):

    def __init__(self, context_dim, act_dim):
        super(Policy, self).__init__()

class CommunicationPolicy(nn.Module):

    def __init__(self, context_dim):
        super(CommunicationPolicy, self).__init__()

class MALNovelD:
    """
    Class for training Multi-Agent Language-NovelD, generating actions and 
    executing training for all agents.
    """
    def __init__(
            self, 
            obs_dim, 
            act_dim, 
            vocab, 
            lr,
            hidden_dim=64, 
            context_dim=64,
            noveld_scale=0.5,
            noveld_trade_off=1):
        self.obs_encoder = ObservationEncoder(obs_dim, context_dim, hidden_dim)
        self.lang_encoder = GRULanguageEncoder(vocab, context_dim)
        self.decoder = GRUDecoder(vocab, context_dim)
        self.policy = Policy(context_dim, act_dim)
        self.comm_policy = CommunicationPolicy(context_dim)
        self.lnoveld = LNovelD(
            obs_dim, context_dim, hidden_dim, noveld_scale, noveld_trade_off)

        self.lr = lr
        self.optimizer = torch.optim.Adam(
            params=self.parameters, lr=self.lr)