import torch
from torch.nn import functional as F

from .networks import MLPNetwork
from .intrinsic_rewards import IntrinsicReward


class E2S_RND(IntrinsicReward):
    """ Elliptical Episodic Scaling of Random Network Distillation. """

    def __init__(self, 
            input_dim, embed_dim, hidden_dim, 
            ridge=0.1, lr=1e-4, device="cpu"):
        self.input_dim = input_dim
        self.ridge = ridge
        self.device = device
        # Random Network Distillation network
        # Fixed target embedding network
        self.target = MLPNetwork(
            input_dim, embed_dim, hidden_dim, n_hidden_layers=3, norm_in=False)
        # Predictor embedding network
        self.predictor = MLPNetwork(
            input_dim, embed_dim, hidden_dim, n_hidden_layers=3, norm_in=False)

        self.device = None

        # Fix weights of target
        for param in self.target.parameters():
            param.requires_grad = False

        # Inverse covariance matrix for Elliptical bonus
        self.ridge = ridge
        self.inv_cov = torch.eye(input_dim).to(device) * (1.0 / self.ridge)
        self.outer_product_buffer = torch.empty(
            input_dim, input_dim).to(device)
        
        # Optimizers
        self.optim = torch.optim.Adam(self.predictor.parameters(), lr=lr)
    
    def init_new_episode(self):
        self.inv_cov = torch.eye(self.input_dim).to(self.device)
        self.inv_cov *= (1.0 / self.ridge)

    def set_train(self, device):
        self.target.train()
        self.target.to(device)
        self.predictor.train()
        self.predictor.to(device)
        self.inv_cov = self.inv_cov.to(device)
        self.outer_product_buffer = self.outer_product_buffer.to(device)
        self.device = device

    def set_eval(self, device):
        self.target.eval()
        self.target.to(device)
        self.predictor.eval()
        self.predictor.to(device)
        self.inv_cov = self.inv_cov.to(device)
        self.outer_product_buffer = self.outer_product_buffer.to(device)
        self.device = device
        
    def get_reward(self, state):
        """
        Get intrinsic reward for the given state.
        Inputs:
            state (torch.Tensor): State from which to generate the reward,
                dim=(1, state_dim).
        Outputs:
            int_reward (float): Intrinsic reward for the input state.
        """
        # Compute embeddings
        target = self.target(state)
        pred = self.predictor(state)

        # Compute novelty
        int_reward = torch.norm(pred.detach() - target.detach(), dim=1, p=2)

        # Compute the elliptic scale
        u = torch.mv(self.inv_cov, state.squeeze())
        elliptic_scale = torch.dot(state, u).item()
        # Update covariance matrix
        torch.outer(u, u, out=self.outer_product_buffer)
        torch.add(
            self.inv_cov, self.outer_product_buffer, 
            alpha=-(1. / (1. + elliptic_scale)), out=self.inv_cov)
        
        return int_reward * elliptic_scale
    
    def train(self, state_batch):
        """
        Inputs:
            state_batch (torch.Tensor): Batch of states, dim=(episode_length, 
                batch_size, state_dim).
        """
        # Encode states
        targets = self.target(state_batch)
        preds = self.predictor(state_batch)
        # Compute loss
        loss = F.mse_loss(preds, targets)
        # Backward pass
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        return float(loss)
    
    def get_params(self):
        return {'target': self.target.state_dict(),
                'predictor': self.predictor.state_dict(),
                'optim': self.optim.state_dict()}

    def load_params(self, params):
        self.target.load_state_dict(params['target'])
        self.predictor.load_state_dict(params['predictor'])
        self.optim.load_state_dict(params['optim'])