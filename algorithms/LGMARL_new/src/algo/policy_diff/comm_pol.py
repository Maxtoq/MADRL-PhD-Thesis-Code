import torch

from torch import nn

from src.algo.nn_modules.mlp import MLPNetwork
from src.algo.nn_modules.rnn import RNNLayer
from src.algo.nn_modules.distributions import DiagGaussian, Categorical
from src.algo.nn_modules.utils import init


def init_(m):
    return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0))


class CP_EmergentDiscrete_Diff(nn.Module):

    def __init__(self, args, parser, device):
        # Comm generation modules
        self.comm_in = MLPNetwork(
            args.hidden_dim, 
            args.context_dim, 
            args.hidden_dim, 
            out_activation_fn="tanh")

        # Discrete language generation and encoding
        

        # Comm encoding modules
