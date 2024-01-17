from torch import nn


class ACCPolicy(nn.Module):

    def __init__(self, args, obs_dim, shared_obs_dim, act_space, device):
        super(ACCPolicy, self).__init__()
        
        # Actor-Communicator
        
        