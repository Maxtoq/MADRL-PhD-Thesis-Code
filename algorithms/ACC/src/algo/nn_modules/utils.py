from torch import nn


def init(module, weight_init=nn.init.orthogonal_, 
         bias_init=lambda x: nn.init.constant_(x, 0), gain=1):
    if hasattr(module, 'weight'):
        weight_init(module.weight.data, gain=gain)
    if hasattr(module, 'bias'):
        bias_init(module.bias.data)
    return module