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
        
        self.ir_mode = args.ir_mode
        self.ir_algo = args.ir_algo
        if self.ir_algo == "e2s_noveld":
            if self.ir_mode == "central":
                obs_dim = get_shape_from_obs_space(shared_obs_space[0])[0]
                act_dim = sum([get_shape_from_act_space(sp) 
                                for sp in act_space])
                print(obs_dim, act_dim)
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

    def get_intrinsic_rewards(self, next_obs_list):
        """
        Get intrinsic reward of the multi-agent system.
        Inputs:
            next_obs_list (list): List of agents' observations at next 
                step.
        Outputs:
            int_rewards (list): List of agents' intrinsic rewards.
        """
        if self.ir_mode == "central":
            # Concatenate observations
            print(next_obs_list)
            cat_obs = torch.Tensor(
                np.concatenate(next_obs_list)).unsqueeze(0).to(self.device)
            print(cat_obs)
            # Get reward
            int_reward = self.ir_model.get_reward(cat_obs)
            int_rewards = [int_reward] * self.nb_agents
        elif self.ir_mode == "local":
            int_rewards = []
            for a_i in range(self.nb_agents):
                obs = torch.Tensor(
                    next_obs_list[a_i]).unsqueeze(0).to(self.device)
                int_rewards.append(self.ir_model[a_i].get_reward(obs))
        return int_rewards

    def prep_rollout(self, device=None):
        super().prep_rollout(device)
        if self.ir_mode == "central":
            self.ir_model.set_eval(device)
        elif self.ir_mode == "local":
            for a_ir in self.ir_model:
                a_ir.set_eval(device)

    def _get_ir_params(self):
        if self.ir_mode == "central":
            return self.int_rew.get_params()
        elif self.ir_mode == "local":
            return [a_int_rew.get_params() for a_int_rew in self.int_rew]

    def _get_save_dict(self):
        save_dict = super()._get_save_dict()
        save_dict["ir_params"] = self._get_ir_params()
        return save_dict

