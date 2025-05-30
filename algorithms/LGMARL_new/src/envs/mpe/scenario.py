import numpy as np
import random

# defines scenario upon which the world is built
class BaseScenario(object):
    # create elements of the world
    def make_world(self):
        raise NotImplementedError()
    # create initial conditions of the world
    def reset_world(self, world):
        raise NotImplementedError()
    def seed(self, seed):
        np.random.seed(seed)
        random.seed(seed)
