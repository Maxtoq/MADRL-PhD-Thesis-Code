import torch

import torch.nn as nn

from src.algo.nn_modules.mlp import MLPNetwork


class Decoder(nn.Module):

    def __init__(self, args, latent_dim, output_dim, device="cpu"):
        """
        Initializes the Decoder of the Autoencoder.

        Args:
            latent_dim (int): Dimension of the latent vector.
            output_dim (int): Dimension of the output vector.
        """
        super(Decoder, self).__init__()
        self.device = device

        self.mlp = MLPNetwork(
            latent_dim,
            output_dim,
            args.hidden_dim,
            args.policy_layer_N)

        self.optim = torch.optim.Adam(
            self.parameters(),
            lr=args.lang_lr)

    def forward(self, x):
        return self.mlp(x)

    def prep_rollout(self, device):
        self.device = device
        self.eval()
        self.to(self.device)

    def prep_training(self, device):
        self.device = device
        self.train()
        self.to(self.device)