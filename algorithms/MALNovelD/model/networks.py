import torch

from torch import nn


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


class MLPNetwork(nn.Module):

    def __init__(self, 
            input_dim, 
            out_dim, 
            hidden_dim=64, 
            n_layers=1, 
            activation_fn='relu'):
        super(MLPNetwork, self).__init__()
        self.n_layers = n_layers

        # Choice for activation function and initialisation of parameters
        if activation_fn not in ['tanh', 'relu']:
            print("ERROR in declaration of MLPNetwork: bad activation_fn with", activation_fn)
        activ_fn = {
            'tanh': nn.Tanh(),
            'relu': nn.ReLU()
        }[activation_fn]
        init_method = nn.init.orthogonal_
        gain = nn.init.calculate_gain(activation_fn)

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), activ_fn
            *[nn.Linear(hidden_dim, hidden_dim), activ_fn]
        )

