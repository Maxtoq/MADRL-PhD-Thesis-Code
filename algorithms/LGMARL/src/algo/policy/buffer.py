


class ACC_ReplayBuffer:

    def __init__(self, 
            args, n_agents, obs_dim, shared_obs_dim, env_act_dim, comm_act_dim):
        self.episode_length = args.episode_length
        self.n_parallel_envs = args.n_parallel_envs
        self.hidden_size = args.hidden_dim
        self.recurrent_N = args.policy_recurrent_N
        self.gamma = args.gamma
        self.gae_lambda = args.gae_lambda
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.shared_obs_dim = shared_obs_dim
        self.env_act_dim = env_act_dim
        self.comm_act_dim = comm_act_dim

        self.obs = np.zeros(
            (self.episode_length + 1, 
             self.n_parallel_envs, 
             n_agents, 
             self.obs_dim),
            dtype=np.float32)
        self.shared_obs = np.zeros(
            (self.episode_length + 1, 
             self.n_parallel_envs, 
             n_agents,
             self.shared_obs_dim),
            dtype=np.float32)

        self.rnn_states = np.zeros(
            (self.episode_length + 1, 
             self.n_parallel_envs, 
             n_agents,
             self.recurrent_N, 
             self.hidden_size), 
            dtype=np.float32)
        self.rnn_states_critic = np.zeros_like(self.rnn_states)

        self.value_preds = np.zeros(
            (self.episode_length + 1, self.n_parallel_envs, n_agents, 1),
            dtype=np.float32)
        self.returns = np.zeros(
            (self.episode_length + 1, self.n_parallel_envs, n_agents, 1),
            dtype=np.float32)

        self.env_actions = np.zeros(
            (self.episode_length, 
             self.n_parallel_envs, 
             n_agents, 
             self.env_act_dim),
            dtype=np.float32)
        self.env_action_log_probs = np.zeros(
            (self.episode_length, 
             self.n_parallel_envs, 
             n_agents, 
             self.env_act_dim),
            dtype=np.float32)

        self.comm_actions = np.zeros(
            (self.episode_length, 
             self.n_parallel_envs, 
             n_agents, 
             self.comm_act_dim),
            dtype=np.float32)
        self.comm_action_log_probs = np.zeros(
            (self.episode_length, 
             self.n_parallel_envs, 
             n_agents, 
             self.comm_act_dim),
            dtype=np.float32)

        self.rewards = np.zeros(
            (self.episode_length, self.n_parallel_envs, n_agents, 1), 
            dtype=np.float32)
        
        self.masks = np.ones(
            (self.episode_length + 1, self.n_parallel_envs, n_agents, 1), 
            dtype=np.float32)

        self.step = 0