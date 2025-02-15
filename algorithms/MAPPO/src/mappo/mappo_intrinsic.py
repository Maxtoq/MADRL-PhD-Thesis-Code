import torch
import numpy as np

from .mappo import MAPPO
from .utils import get_shape_from_obs_space, get_shape_from_act_space
from ..intrinsic_rewards.e2s_noveld import E2S_NovelD

def get_ir_class_params(args):
    pass


class MAPPO_IR(MAPPO):
    """
    Class implementing MAPPO with Intrinsic Rewards.
    """
    def __init__(self, 
            args, n_agents, obs_space, shared_obs_space, act_space, device):
        super(MAPPO_IR, self).__init__(
            args, n_agents, obs_space, shared_obs_space, act_space, device)
        self.device = device
        self.ir_mode = args.ir_mode
        self.ir_algo = args.ir_algo
        if self.ir_algo == "e2s_noveld":
            if self.ir_mode == "central":
                obs_dim = get_shape_from_obs_space(shared_obs_space[0])[0]
                act_dim = sum([get_shape_from_act_space(sp) 
                                for sp in act_space])
                self.ir_model = E2S_NovelD(
                    obs_dim, 
                    act_dim,
                    args.ir_enc_dim, 
                    args.ir_hidden_dim, 
                    args.ir_scale_fac,
                    args.ir_ridge,
                    args.ir_lr, 
                    device,
                    args.ir_ablation)
            elif self.ir_mode == "local":
                self.ir_model = [
                    E2S_NovelD(
                        get_shape_from_obs_space(obs_space[a_i])[0], 
                        get_shape_from_act_space(act_space[a_i]),
                        args.ir_enc_dim, 
                        args.ir_hidden_dim, 
                        args.ir_scale_fac,
                        args.ir_ridge,
                        args.ir_lr, 
                        device,
                        args.ir_ablation)
                    for a_i in range(self.n_agents)]
        else:
            print("Wrong intrinsic reward algo")
            raise NotImplementedError

    def start_episode(self, obs, n_episodes=1):
        super().start_episode(obs, n_episodes)
        if self.ir_mode == "central":
            self.ir_model.init_new_episode(n_episodes)
            # Initialise intrinsic reward model with first observation
            share_obs = obs.reshape(obs.shape[0], -1)
            self.ir_model.get_reward(torch.Tensor(share_obs).to(self.device))
        elif self.ir_mode == "local":
            # Reshape observations by agents
            obs = torch.Tensor(obs.transpose((1, 0, 2))).to(self.device)
            for a_i in range(self.n_agents):
                self.ir_model[a_i].init_new_episode(n_episodes)
                self.ir_model[a_i].get_reward(obs[a_i])

    def get_intrinsic_rewards(self, next_obs):
        """
        Get intrinsic reward of the multi-agent system.
        Inputs:
            next_obs (numpy.ndarray): Next observations, 
                dim=(batch_size, n_agents, obs_dim).
        Outputs:
            intr_rewards (numpy.ndarray): Intrinsic rewards, 
                dim=(batch_size, n_agents).
        """
        if self.ir_mode == "central":
            # Concatenate observations
            next_share_obs = next_obs.reshape(next_obs.shape[0], -1)
            # Get reward
            int_rewards = self.ir_model.get_reward(
                torch.Tensor(next_share_obs).to(self.device))
            intr_rewards = int_rewards.unsqueeze(-1).repeat(1, self.n_agents)
        elif self.ir_mode == "local":
            intr_rewards = []
            # Reshape observations by agents
            next_obs = torch.Tensor(
                next_obs.transpose((1, 0, 2))).to(self.device)
            for a_i in range(self.n_agents):
                intr_rewards.append(
                    self.ir_model[a_i].get_reward(next_obs[a_i]))
            intr_rewards = torch.stack(intr_rewards, dim=1)
        return intr_rewards.cpu().numpy()

    def train(self):
        mappo_losses = super().train()

        if self.ir_mode == "central":
            share_obs = torch.Tensor(self.buffer[0].share_obs).to(self.device)
            share_acts = torch.Tensor(np.concatenate(
                [buff.actions for buff in self.buffer], axis=-1)).to(self.device)
            ir_losses = self.ir_model.train(share_obs, share_acts)
        elif self.ir_mode == "local":
            losses = [
                self.ir_model[a_i].train(
                    torch.Tensor(self.buffer[a_i].obs).to(self.device),
                    torch.Tensor(self.buffer[a_i].actions).to(self.device))
                for a_i in range(self.n_agents)]
            ir_losses = {
                "rnd_loss": np.mean([l["rnd_loss"] for l in losses]),
                "e3b_loss": np.mean([l["e3b_loss"] for l in losses])}
        return mappo_losses, ir_losses

    def prep_rollout(self, device=None):
        if device is None:
            device = self.device
        super().prep_rollout(device)
        self.device = device
        if self.ir_mode == "central":
            self.ir_model.set_eval(device)
        elif self.ir_mode == "local":
            for a_ir in self.ir_model:
                a_ir.set_eval(device)
    
    def prep_training(self, device=None):
        if device is None:
            device = self.train_device
        super().prep_training()
        self.device = device
        if self.ir_mode == "central":
            self.ir_model.set_train(device)
        elif self.ir_mode == "local":
            for a_ir in self.ir_model:
                a_ir.set_train(device)

    def _get_ir_params(self):
        if self.ir_mode == "central":
            return self.ir_model.get_params()
        elif self.ir_mode == "local":
            return [a_ir.get_params() for a_ir in self.ir_model]

    def _get_save_dict(self):
        save_dict = super()._get_save_dict()
        save_dict["ir_params"] = self._get_ir_params()
        return save_dict

