from onpolicy.algorithms.r_mappo.r_mappo import R_MAPPO as TrainAlgo
from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy as Policy


class MAPPO():

    def __init__(self, cfg, obs_dim, act_dim):
        self.args = cfg
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        # Set variant
        if self.args.algorithm_name == "rmappo":
            self.args.use_recurrent_policy = True
            self.args.use_naive_recurrent_policy = False
        elif self.args.algorithm_name == "mappo":
            self.args.use_recurrent_policy = False 
            self.args.use_naive_recurrent_policy = False
        elif self.args.algorithm_name == "ippo":
            self.args.use_centralized_V = False
        else:
            raise NotImplementedError

        # Init agent policies
        self.policy = []
        for agent_id in range(self.num_agents):
            share_observation_space = self.envs.share_observation_space[agent_id] if self.use_centralized_V else self.envs.observation_space[agent_id]
            # policy network
            po = Policy(self.args,
                        self.envs.observation_space[agent_id],
                        share_observation_space,
                        self.envs.action_space[agent_id],
                        device = self.device)
            self.policy.append(po)

        self.trainer = []
        self.buffer = []
        for agent_id in range(self.num_agents):
            # algorithm
            tr = TrainAlgo(self.args, self.policy[agent_id], device = self.device)
            # buffer
            share_observation_space = self.envs.share_observation_space[agent_id] if self.use_centralized_V else self.envs.observation_space[agent_id]
            bu = SeparatedReplayBuffer(self.cfg,
                                       self.envs.observation_space[agent_id],
                                       share_observation_space,
                                       self.envs.action_space[agent_id])
        