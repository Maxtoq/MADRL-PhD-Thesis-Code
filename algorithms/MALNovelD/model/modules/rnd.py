import torch

from torch import nn
from torch.optim import Adam

from .networks import MLPNetwork


class RND:
    """
    Class implementing Random Network Distillation (Burda et al., 2019)
    """
    def __init__(self, state_dim, embed_dim, hidden_dim, lr=1e-4):
        """
        Inputs:
            :param state_dim (int): Dimension of the input.
            :param embed_dim (int): Dimension of the output of RND networks.
            :param hidden_dim (int): Dimension of the hidden layers in MLPs.
            :param lr (float): Learning rate for training the predictor
                (default=0.0001).
        """
        # Random Network Distillation network
        # Fixed target embedding network
        self.target = MLPNetwork(
            state_dim, embed_dim, hidden_dim, n_hidden_layers=3, norm_in=False)
        # Predictor embedding network
        self.predictor = MLPNetwork(
            state_dim, embed_dim, hidden_dim, n_hidden_layers=3, norm_in=False)

        self.device = None

        # Fix weights of target
        for param in self.target.parameters():
            param.requires_grad = False

        # Training objects
        self.lr = lr
        self.predictor_loss = nn.MSELoss()
        self.optim = Adam(self.predictor.parameters(), lr=lr)

        # Stored predictions for future training
        self.stored_preds = torch.Tensor()
        self.stored_targets = torch.Tensor()

    def set_train(self, device):
        self.target.train()
        self.target.to(device)
        self.predictor.train()
        self.predictor.to(device)
        self.stored_preds = self.stored_preds.to(device)
        self.stored_targets = self.stored_targets.to(device)
        self.device = device

    def set_eval(self, device):
        self.target.eval()
        self.target.to(device)
        self.predictor.eval()
        self.predictor.to(device)
        self.stored_preds = self.stored_preds.to(device)
        self.stored_targets = self.stored_targets.to(device)
        self.device = device
    
    def store_pred(self, pred, target):
        self.stored_preds = torch.cat(
            (self.stored_preds, pred)).to(self.device)
        self.stored_targets = torch.cat(
            (self.stored_targets, target)).to(self.device)

    def train_predictor(self):
        self.optim.zero_grad()
        loss = self.predictor_loss(self.stored_preds, self.stored_targets)
        loss.backward()
        self.optim.step()

        self.stored_preds = torch.Tensor().to(self.device)
        self.stored_targets = torch.Tensor().to(self.device)

        return float(loss)

    def get_reward(self, state):
        """
        Get intrinsic reward for this new state.
        Inputs:
            state (torch.Tensor): State from which to generate the reward,
                dim=(1, state_dim).
        Outputs:
            nov (float): Intrinsic reward for the input state.
        """
        # Compute embeddings
        target = self.target(state)
        pred = self.predictor(state)

        # Compute novelty
        int_reward = torch.norm(pred.detach() - target.detach(), dim=1, p=2)

        # Store predictions
        self.store_pred(pred, target)

        return int_reward

    def get_params(self):
        return {'target': self.target.state_dict(),
                'predictor': self.predictor.state_dict(),
                'optim': self.optim.state_dict()}