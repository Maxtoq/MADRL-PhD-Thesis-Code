import numpy as np

from gym.spaces import Box, Discrete, Tuple


def get_dim_from_space(space):
    if isinstance(space, Box):
        dim = space.shape[0]
    elif isinstance(space, Discrete):
        dim = space.n
    else:
        raise Exception("Unrecognized space: ", type(space))
    return dim

def get_cent_act_dim(action_space):
    cent_act_dim = 0
    for space in action_space:
        dim = get_dim_from_space(space)
        if isinstance(dim, np.ndarray):
            cent_act_dim += int(sum(dim))
        else:
            cent_act_dim += dim
    return cent_act_dim