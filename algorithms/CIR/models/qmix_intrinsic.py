import torch
import numpy as np

from .modules.qmix import QMIX
from .modules.lnoveld import NovelD
from .modules.e3b import E3B

class QMIX_CIR(QMIX):
    """ 
    Class impelementing QMIX with Centralised Intrinsic Rewards (QMIX_CIR), 
    meaning that we use a single intrinsic reward generation model to compute 
    the intrinsic reward of the multi-agent system.
    """
    def __init__(self, nb_agents, obs_dim, act_dim, lr, 
                 gamma=0.99, tau=0.01, hidden_dim=64, shared_params=False, 
                 init_explo_rate=1.0, max_grad_norm=None, device="cpu",
                 intrinsic_reward_algo="none", intrinsic_reward_params={}):
                #  embed_dim=16, nd_lr=1e-4, nd_scale_fac=0.5, nd_hidden_dim=64):
        super(QMIX_CIR, self).__init__(
            nb_agents, obs_dim, act_dim, lr, gamma, tau, hidden_dim, 
            shared_params, init_explo_rate, max_grad_norm, device)

        self.intrinsic_reward_algo = intrinsic_reward_algo
        self.intrinsic_reward_params = intrinsic_reward_params
        if intrinsic_reward_algo == "none":
            self.cent_int_rew = None
        elif intrinsic_reward_algo == 'cent_noveld':
            self.cent_int_rew = NovelD(**intrinsic_reward_params)
        elif intrinsic_reward_algo == 'cent_e3b':
            self.cent_int_rew = E3B(**intrinsic_reward_params)
        else:
            print("ERROR: wrong intrinsic reward algorithm, must be in ['none', 'noveld', 'e3b'].")
            exit()

    def get_intrinsic_rewards(self, next_obs_list):
        """
        Get intrinsic reward of the multi-agent system.
        Inputs:
            next_obs_list (list): List of agents' observations at next 
                step.
        Outputs:
            int_rewards (list): List of agents' intrinsic rewards.
        """
        if self.cent_int_rew is None:
            return [0.0] * self.nb_agents
        else:
            # Concatenate observations
            cat_obs = torch.Tensor(
                np.concatenate(next_obs_list)).unsqueeze(0).to(self.device)
            # Get reward
            int_reward = self.cent_int_rew.get_reward(cat_obs)
            int_rewards = [int_reward] * self.nb_agents
            return int_rewards
    
    def train(self, batch):
        """
        Update all agents and Intrinsic reward model.
        Inputs:
            batch (tuple(torch.Tensor)): Tuple of batches of experiences for
                the agents to train on.
        Outputs:
            qtot_loss (float): QMIX loss.
            int_rew_loss (float): Intrinsic reward loss.
        """
        qtot_loss = super().train_on_batch(batch)

        # Intrinsic reward model update
        if self.cent_int_rew is None:
            int_rew_loss = 0.0
        else:
            _, shared_obs_b, act_b, _, _ = batch
            act_b = torch.cat(tuple(act_b), dim=-1)
            int_rew_loss = self.cent_int_rew.train(shared_obs_b[0], act_b)

        return qtot_loss, float(int_rew_loss)

    def reset_int_reward(self, obs_list):
        if self.cent_int_rew is not None:
            # Reset intrinsic reward model
            self.cent_int_rew.init_new_episode()
            # Initialise intrinsic reward model with first observation
            cat_obs = torch.Tensor(
                np.concatenate(obs_list)).unsqueeze(0).to(self.device)
            self.cent_int_rew.get_reward(cat_obs.view(1, -1))
    
    def prep_training(self, device='cpu'):
        super().prep_training(device)
        if self.cent_int_rew is not None:
            self.cent_int_rew.set_train(device)
    
    def prep_rollouts(self, device='cpu'):
        super().prep_rollouts(device)
        if self.cent_int_rew is not None:
            self.cent_int_rew.set_eval(device)

    def save(self, filename):
        self.prep_training(device='cpu')
        save_dict = {
            'nb_agents': self.nb_agents,
            'obs_dim': self.obs_dim,
            'act_dim': self.act_dim,
            'lr': self.lr,
            'gamma': self.gamma,
            'tau': self.tau,
            'hidden_dim': self.hidden_dim,
            'shared_params': self.shared_params,
            'max_grad_norm': self.max_grad_norm,
            "intrinsic_reward_algo": self.intrinsic_reward_algo, 
            "intrinsic_reward_params": self.intrinsic_reward_params,
            'agent_params': [a.get_params() for a in self.agents],
            'mixer_params': self.mixer.state_dict(),
            'target_mixer_params': self.target_mixer.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        if self.cent_int_rew is not None:
            save_dict['cent_int_rew_params'] = self.cent_int_rew.get_params()
        torch.save(save_dict, filename)

    @classmethod
    def init_from_save(cls, filename):
        """
        Instantiate instance of this class from file created by 'save' method
        """
        save_dict = torch.load(filename, map_location=torch.device('cpu'))
        agent_params = save_dict.pop("agent_params")
        mixer_params = save_dict.pop("mixer_params")
        target_mixer_params = save_dict.pop("target_mixer_params")
        optimizer = save_dict.pop("optimizer")
        if "cent_int_rew_params" in save_dict:
            cent_int_rew_params = save_dict.pop("cent_int_rew_params")
        instance = cls(**save_dict)
        for a, params in zip(instance.agents, agent_params):
            a.load_params(params)
        instance.mixer.load_state_dict(mixer_params)
        instance.target_mixer.load_state_dict(target_mixer_params)
        instance.optimizer.load_state_dict(optimizer)
        if instance.cent_int_rew is not None:
            instance.cent_int_rew.load_params(cent_int_rew_params)
        return instance