import torch
import torch.nn as nn
import numpy as np
from torch.distributions import OneHotCategorical
from offpolicy.algorithms.base.mlp_policy import MLPPolicy
from offpolicy.algorithms.utils.mlp import MLPBase
from offpolicy.algorithms.utils.act import ACTLayer

from offpolicy.utils.util import init, to_torch
from offpolicy.utils.util import is_discrete, is_multidiscrete, init, to_torch,\
    get_dim_from_space, DecayThenFlatSchedule, soft_update, hard_update, \
    gumbel_softmax, onehot_from_logits, gaussian_noise, avail_choose, to_numpy


class MADDPG_Actor(nn.Module):
    def __init__(self, args, obs_dim, act_dim, device):
        """
        Actor class for MADDPG/MATD3. Outputs actions given observations.
        :param args: (argparse.Namespace) arguments containing relevant model 
                     information.
        :param obs_dim: (int) dimension of the observation vector.
        :param act_dim: (int) dimension of the action vector.
        :param device: (torch.device) specifies the device to run on (cpu/gpu).
        """
        super(MADDPG_Actor, self).__init__()
        self._use_orthogonal = args.use_orthogonal
        self._gain = args.gain
        self.hidden_size = args.hidden_size
        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)

        # map observation input into input for rnn
        self.mlp = MLPBase(args, obs_dim)

        # get action from rnn hidden state
        self.act = ACTLayer(
            act_dim, self.hidden_size, self._use_orthogonal, self._gain)

        self.to(device)

    def forward(self, x):
        """
        Compute actions using the needed information.
        :param x: (np.ndarray) Observations with which to compute actions.
        """
        x = to_torch(x).to(**self.tpdv)
        x = self.mlp(x)
        # pass outputs through linear layer
        action = self.act(x)

        return action


class MADDPG_Critic(nn.Module):
    """
    Critic network class for MADDPG/MATD3. Outputs actions given observations.
    :param args: (argparse.Namespace) arguments containing relevant model information.
    :param central_obs_dim: (int) dimension of the centralized observation vector.
    :param central_act_dim: (int) dimension of the centralized action vector.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    :param num_q_outs: (int) number of q values to output (1 for MADDPG, 2 for MATD3).
    """
    def __init__(self, args, central_obs_dim, central_act_dim, device, num_q_outs=1):
        super(MADDPG_Critic, self).__init__()
        self._use_orthogonal = args.use_orthogonal
        self.hidden_size = args.hidden_size
        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)

        input_dim = central_obs_dim + central_act_dim

        self.mlp = MLPBase(args, input_dim)

        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][self._use_orthogonal]
        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))
        self.q_outs = [init_(nn.Linear(self.hidden_size, 1).to(device)) for _ in range(num_q_outs)]
        
        self.to(device)

    def forward(self, central_obs, central_act):
        """
        Compute Q-values using the needed information.
        :param central_obs: (np.ndarray) Centralized observations with which to compute Q-values.
        :param central_act: (np.ndarray) Centralized actions with which to compute Q-values.

        :return q_values: (list) Q-values outputted by each Q-network.
        """
        central_obs = to_torch(central_obs).to(**self.tpdv)
        central_act = to_torch(central_act).to(**self.tpdv)

        x = torch.cat([central_obs, central_act], dim=1)

        x = self.mlp(x)
        q_values = [q_out(x) for q_out in self.q_outs]

        return q_values


class MATD3_Actor(MADDPG_Actor):
    """MATD3 Actor is identical to MADDPG Actor, see parent class"""
    pass

class MATD3_Critic(MADDPG_Critic):
    """MATD3 Critic class. Identical to MADDPG Critic, but with 2 Q output.s"""
    def __init__(self, args, central_obs_dim, central_act_dim, device):
        super(MATD3_Critic, self).__init__(args, central_obs_dim, central_act_dim, device, num_q_outs=2)



class MADDPGPolicy(MLPPolicy):
    """
    MADDPG/MATD3 Policy Class to wrap actor/critic and compute actions. See parent class for details.
    :param config: (dict) contains information about hyperparameters and algorithm configuration
    :param policy_config: (dict) contains information specific to the policy (obs dim, act dim, etc)
    :param target_noise: (int) std of target smoothing noise to add for MATD3 (applies only for continuous actions)
    :param td3: (bool) whether to use MATD3 or MADDPG.
    :param train: (bool) whether the policy will be trained.
    """
    def __init__(self, config, policy_config, target_noise=None, td3=False, train=True):
        self.config = config
        self.device = config['device']
        self.args = self.config["args"]
        self.tau = self.args.tau
        self.lr = self.args.lr
        self.opti_eps = self.args.opti_eps
        self.weight_decay = self.args.weight_decay

        self.central_obs_dim, self.central_act_dim = policy_config[ "cent_obs_dim"], policy_config["cent_act_dim"]
        self.obs_space = policy_config["obs_space"]
        self.obs_dim = get_dim_from_space(self.obs_space)
        self.act_space = policy_config["act_space"]
        self.discrete = is_discrete(self.act_space)
        self.multidiscrete = is_multidiscrete(self.act_space)

        self.act_dim = get_dim_from_space(self.act_space)
        self.output_dim = sum(self.act_dim) if isinstance(self.act_dim, np.ndarray) else self.act_dim
        self.target_noise = target_noise

        actor_class = MATD3_Actor if td3 else MADDPG_Actor
        critic_class = MATD3_Critic if td3 else MADDPG_Critic

        self.actor = actor_class(self.args, self.obs_dim, self.act_dim, self.device)
        self.critic = critic_class(self.args, self.central_obs_dim, self.central_act_dim, self.device)

        self.target_actor = actor_class(self.args, self.obs_dim, self.act_dim, self.device)
        self.target_critic = critic_class(self.args, self.central_obs_dim, self.central_act_dim, self.device)

        # sync the target weights
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        if train:
            self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr, eps=self.opti_eps, weight_decay=self.weight_decay)
            self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr, eps=self.opti_eps, weight_decay=self.weight_decay)

            if self.discrete:
                # eps greedy exploration
                self.exploration = DecayThenFlatSchedule(self.args.epsilon_start, self.args.epsilon_finish,
                                                         self.args.epsilon_anneal_time, decay="linear")


    def get_actions(self, obs, available_actions=None, t_env=None, explore=False, use_target=False, use_gumbel=False):
        """See parent class."""
        batch_size = obs.shape[0]
        eps = None
        if use_target:
            actor_out = self.target_actor(obs)
        else:
            actor_out = self.actor(obs)

        if self.discrete:
            if self.multidiscrete:
                if use_gumbel or (use_target and self.target_noise is not None):
                    onehot_actions = list(map(lambda a: gumbel_softmax(a, hard=True, device=self.device), actor_out))
                    actions = torch.cat(onehot_actions, dim=-1)
                elif explore:
                    onehot_actions = list(map(lambda a: gumbel_softmax(a, hard=True, device=self.device), actor_out))
                    onehot_actions = torch.cat(onehot_actions, dim=-1)
                    # eps greedy exploration
                    eps = self.exploration.eval(t_env)
                    rand_numbers = np.random.rand(batch_size, 1)
                    take_random = (rand_numbers < eps).astype(int).reshape(-1, 1)
                    # random actions sample uniformly from action space
                    random_actions = [OneHotCategorical(logits=torch.ones(batch_size, self.act_dim[i])).sample() for i in range(len(self.act_dim))]
                    random_actions = torch.cat(random_actions, dim=1)
                    actions = (1 - take_random) * to_numpy(onehot_actions) + take_random * to_numpy(random_actions)
                else:
                    onehot_actions = list(map(onehot_from_logits, actor_out))
                    actions = torch.cat(onehot_actions, dim=-1)
  
            else:
                if use_gumbel or (use_target and self.target_noise is not None):
                    actions = gumbel_softmax(actor_out, available_actions, hard=True, device=self.device)  # gumbel has a gradient 
                elif explore:
                    onehot_actions = gumbel_softmax(actor_out, available_actions, hard=True, device=self.device)  # gumbel has a gradient                    
                    # eps greedy exploration
                    eps = self.exploration.eval(t_env)
                    rand_numbers = np.random.rand(batch_size, 1)
                    # random actions sample uniformly from action space
                    logits = avail_choose(torch.ones(batch_size, self.act_dim), available_actions)
                    random_actions = OneHotCategorical(logits=logits).sample().numpy()
                    take_random = (rand_numbers < eps).astype(int)
                    actions = (1 - take_random) * to_numpy(onehot_actions) + take_random * random_actions
                else:
                    actions = onehot_from_logits(actor_out, available_actions)  # no gradient

        else:
            if explore:
                actions = gaussian_noise(actor_out.shape, self.args.act_noise_std) + actor_out
            elif use_target and self.target_noise is not None:
                assert isinstance(self.target_noise, float)
                actions = gaussian_noise(actor_out.shape, self.target_noise) + actor_out
            else:
                actions = actor_out
            # # clip the actions at the bounds of the action space
            # actions = torch.max(torch.min(actions, torch.from_numpy(self.act_space.high)), torch.from_numpy(self.act_space.low))

        return actions, eps

    def get_random_actions(self, obs, available_actions=None):
        """See parent class."""
        batch_size = obs.shape[0]

        if self.discrete:
            if self.multidiscrete:
                random_actions = [OneHotCategorical(logits=torch.ones(batch_size, self.act_dim[i])).sample().numpy() for i in
                                    range(len(self.act_dim))]
                random_actions = np.concatenate(random_actions, axis=-1)
            else:
                if available_actions is not None:
                    logits = avail_choose(torch.ones(batch_size, self.act_dim), available_actions)
                    random_actions = OneHotCategorical(logits=logits).sample().numpy()
                else:
                    random_actions = OneHotCategorical(logits=torch.ones(batch_size, self.act_dim)).sample().numpy()
        else:
            random_actions = np.random.uniform(self.act_space.low, self.act_space.high, size=(batch_size, self.act_dim))

        return random_actions

    def soft_target_updates(self):
        """Polyal update the target networks."""
        # polyak updates to target networks
        soft_update(self.target_critic, self.critic, self.args.tau)
        soft_update(self.target_actor, self.actor, self.args.tau)

    def hard_target_updates(self):
        """Copy the live networks into the target networks."""
        # polyak updates to target networks
        hard_update(self.target_critic, self.critic)
        hard_update(self.target_actor, self.actor)
    
    def save_actor_state(self, cp_path):
        self.actor.to(torch.device('cpu'))
        torch.save(self.actor.state_dict(), cp_path)
        self.actor.to(self.device)

    def load_actor_state(self, cp_path):
        self.actor.load_state_dict(torch.load(cp_path))
        self.actor.to(self.device)


class MATD3Policy(MADDPGPolicy):
    def __init__(self, config, policy_config, train=True):
        """See parent class."""
        super(MATD3Policy, self).__init__(config, policy_config, target_noise=config["args"].target_action_noise_std, td3=True, train=train)