import copy
import logging

import gym
import numpy as np
from PIL import ImageColor
from gym import spaces
from gym.utils import seeding

from src.envs.ma_gym.utils.action_space import MultiAgentActionSpace
from src.envs.ma_gym.utils.draw import draw_grid, fill_cell, draw_circle, write_cell_text
from src.envs.ma_gym.utils.observation_space import MultiAgentObservationSpace

logger = logging.getLogger(__name__)


class EmptyEnv(gym.Env):
    """
    Empty environment, with one agent in the center (multiple agents can be handled, 
    but with no interaction between them), made for testing and evaluation purposes.
    """

    def __init__(self, grid_shape=(5, 5), n_agents=2, max_steps=100, agent_view_mask=(5, 5)):
        self._grid_shape = grid_shape
        self.n_agents = n_agents
        self._max_steps = max_steps
        self._agent_view_mask = agent_view_mask

        self.action_space = MultiAgentActionSpace([spaces.Discrete(5) for _ in range(self.n_agents)])
        self.agent_pos = {_: None for _ in range(self.n_agents)}

        self._base_grid = self.__create_grid()  # with no agents
        self._full_obs = self.__create_grid()

        self._agent_dones = [False for _ in range(self.n_agents)]
        self.viewer = None

        mask_size = np.prod(self._agent_view_mask)
        self._obs_high = np.ones(2 + mask_size, dtype=np.float32)
        self._obs_low = np.zeros(2 + mask_size, dtype=np.float32)