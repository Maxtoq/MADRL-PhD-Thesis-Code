import torch

from torch import nn


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

class LNovelD(nn.Module):
    """
    Class implementing the Language-augmented version of NovelD from Mu et al. 
    (2022).
    :param obs_in_dim (int): Dimension of the observation input
    :param lang_in_dim (int): Dimension of the language encoding input
    :param hidden_dim (int): Dimension of the hidden layers in MLPs
    :param scale_fac (float): Scaling factor for computing the reward, noted 
        alpha in the paper, controls how novel we want the states to be to 
        generate some reward (in [0,1])
    :param trade_off (float): Parameter for controlling the weight of the 
        language novelty in the final reward, noted lambda_l in the paper (in 
        [0, +inf])
    """
    def __init__(self, 
            obs_in_dim, 
            lang_in_dim, 
            hidden_dim=64, 
            scale_fac=0.5, 
            trade_off=1):
        super(LNovelD, self).__init__()
        # Fixed random target embedding network
        


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