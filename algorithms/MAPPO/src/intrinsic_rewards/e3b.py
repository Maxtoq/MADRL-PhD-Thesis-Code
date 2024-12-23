import torch
from torch.nn import functional as F

from .networks import MLPNetwork
from .intrinsic_rewards import IntrinsicReward


class E3B(IntrinsicReward):
    
    def __init__(self, input_dim, act_dim, enc_dim, 
            hidden_dim=64, ridge=0.1, lr=1e-4, device="cpu"):
        self.enc_dim = enc_dim
        self.ridge = ridge
        self.device = device
        # State encoder
        self.encoder = MLPNetwork(
            input_dim, enc_dim, hidden_dim, norm_in=False)
        # Inverse dynamics model
        self.inv_dyn = MLPNetwork(
            2 * enc_dim, act_dim, hidden_dim, norm_in=False)
        # Inverse covariance matrix
        self.ridge = ridge
        self.inv_cov = torch.cat([
            torch.eye(self.enc_dim).unsqueeze(0) for _ in range(1)])
        self.outer_product_buffer = torch.empty(enc_dim, enc_dim).to(device)
        
        # Optimizers
        self.encoder_optim = torch.optim.Adam(
            self.encoder.parameters(), 
            lr=lr)
        self.inv_dyn_optim = torch.optim.Adam(
            self.inv_dyn.parameters(), 
            lr=lr)
    
    def init_new_episode(self, n_episodes=1):
        self.inv_cov = torch.cat([
            torch.eye(self.enc_dim).unsqueeze(0) for _ in range(n_episodes)
        ]).to(self.device)
        self.inv_cov *= (1.0 / self.ridge)

    def set_train(self, device):
        self.encoder.train()
        self.encoder = self.encoder.to(device)
        self.inv_dyn.train()
        self.inv_dyn = self.inv_dyn.to(device)
        self.inv_cov = self.inv_cov.to(device)
        self.outer_product_buffer = self.outer_product_buffer.to(device)
        self.device = device

    def set_eval(self, device):
        self.encoder.eval()
        self.encoder = self.encoder.to(device)
        self.inv_cov = self.inv_cov.to(device)
        self.outer_product_buffer = self.outer_product_buffer.to(device)
        self.device = device
        
    def get_reward(self, state_batch):
        """
        Inputs:
            state_batch (torch.Tensor): dim=(batch_size, state_dim)
        """
        # Encode state
        enc_state = self.encoder(state_batch).detach().unsqueeze(-1)
        # Compute the intrinsic reward
        u = torch.matmul(enc_state.transpose(1, 2), self.inv_cov)
        int_rewards = torch.matmul(u, enc_state).squeeze()
        # Update covariance matrix
        outer_prod = torch.matmul(u.transpose(1, 2), u)
        batch_size = state_batch.shape[0]
        if batch_size > 1:
            self.inv_cov = torch.cat([
                torch.add(self.inv_cov[e_i], outer_prod[e_i], 
                    alpha=-(1. / (1. + int_rewards[e_i]))).unsqueeze(0)
                for e_i in range(batch_size)
            ])
        else:
            torch.add(
                self.inv_cov, outer_prod, 
                alpha=-(1. / (1. + int_rewards)), out=self.inv_cov)
        return int_rewards
    
    def train(self, state_batch, act_batch):
        """
        Inputs:
            state_batch (torch.Tensor): Batch of states, dim=(episode_length + 1, 
                batch_size, state_dim).
            act_batch (torch.Tensor): Batch of actions, dim=(episode_length, 
                batch_size, action_dim).
        """
        # Encode states
        enc_all_states_b = self.encoder(state_batch)
        enc_states_b = enc_all_states_b[:-1]
        enc_next_states_b = enc_all_states_b[1:]
        # Run inverse dynamics model
        inv_dyn_inputs = torch.cat((enc_states_b, enc_next_states_b), dim=-1)
        pred_actions = self.inv_dyn(inv_dyn_inputs)
        # Compute loss
        loss = F.mse_loss(pred_actions, act_batch)
        # Backward pass
        self.encoder_optim.zero_grad()
        self.inv_dyn_optim.zero_grad()
        loss.backward()
        self.encoder_optim.step()
        self.inv_dyn_optim.step()
        return float(loss)
    
    def get_params(self):
        return {'encoder': self.encoder.state_dict(),
                'inv_dyn': self.inv_dyn.state_dict(),
                'encoder_optim': self.encoder_optim.state_dict(),
                'inv_dyn_optim': self.inv_dyn_optim.state_dict()}

    def load_params(self, params):
        self.encoder.load_state_dict(params['encoder'])
        self.inv_dyn.load_state_dict(params['inv_dyn'])
        self.encoder_optim.load_state_dict(params['encoder_optim'])
        self.inv_dyn_optim.load_state_dict(params['inv_dyn_optim'])