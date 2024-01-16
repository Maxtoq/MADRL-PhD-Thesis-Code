


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

        self.policy = ACCPolicy(args, obs_dim, shared_obs_dim, act_space, device)

