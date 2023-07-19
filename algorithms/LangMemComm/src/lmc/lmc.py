from modules.lm import OneHotEncoder, GRUEncoder, GRUDecoder
from modules.obs import ObservationEncoder
from policy.mappo import MAPPO


class LMC:
    """
    Language-Memory for Communication using a pre-defined discrete language.
    """
    def __init__(self, args, n_agents, 
                 obs_dim, shared_obs_dim, act_dim, 
                 vocab, device):
        self.args = args
        self.n_agents = n_agents

        # Modules
        self.obs_encoder = ObservationEncoder(
            obs_dim, args.context_dim, args.hidden_dim)
        self.word_encoder = OneHotEncoder(vocab)
        self.sentence_encoder = GRUEncoder(context_dim, self.word_encoder)
        self.decoder = GRUDecoder(context_dim, self.word_encoder)
        # self.comm_policy = CommunicationPolicy(context_dim, hidden_dim)
        if self.policy_algo == "mappo":
            self.policy = MAPPO(
                cfg, n_agents, obs_space, shared_obs_space, act_space, device)
