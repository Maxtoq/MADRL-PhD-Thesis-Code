import torch

from .acc_policy import ACCPolicy
from .utils import update_linear_schedule, update_lr


class CommNActPolicy:

    def __init__(self, args, lang_learner, n_agents, obs_space, 
                 shared_obs_space, act_space, device):
        self.args = args
        self.lang_learner = lang_learner
        self.n_agents = n_agents
        self.device = device
        self.n_parallel_envs = args.n_parallel_envs
        self.recurrent_N = args.recurrent_N
        self.hidden_dim = args.hidden_size
        self.lr = args.lr
        self.warming_up = False

        self.policy = ACCPolicy(args, obs_dim, shared_obs_dim, act_space, device)

        self.rl_optim = torch.optim.Adam(
            self.policy.parameters(), 
            lr=self.lr, 
            eps=args.opti_eps, 
            weight_decay=args.weight_decay)

    def lr_decay(self, episode, episodes):
        """
        Decay the actor and critic learning rates.
        :param episode: (int) current training episode.
        :param episodes: (int) total number of training episodes.
        """
        update_linear_schedule(
            self.rl_optim, episode, episodes, self.lr)

    def warmup_lr(self, warmup):
        if warmup != self.warming_up:
            lr = self.lr * 0.01 if warmup else self.lr
            update_lr(self.rl_optim, lr)
            self.warming_up = warmup

