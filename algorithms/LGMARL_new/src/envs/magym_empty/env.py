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

    def __init__(self, grid_shape=(5, 5), n_agents=2, max_steps=100, agent_view_mask=(5, 5),
                 see_agents=False):
        self._grid_shape = grid_shape
        self.n_agents = n_agents
        self._max_steps = max_steps
        self._agent_view_mask = agent_view_mask
        self._step_count = None
        self._see_agents = see_agents

        self.action_space = MultiAgentActionSpace([spaces.Discrete(5) for _ in range(self.n_agents)])
        self.agent_pos = {_: None for _ in range(self.n_agents)}

        self._base_grid = self.__create_grid()  # with no agents
        self._full_obs = self.__create_grid()

        self._agent_dones = [False for _ in range(self.n_agents)]
        self.viewer = None

        mask_size = np.prod(self._agent_view_mask)
        self._obs_high = np.ones(2 + mask_size, dtype=np.float32)
        self._obs_low = np.zeros(2 + mask_size, dtype=np.float32)
        self.observation_space = MultiAgentObservationSpace(
            [spaces.Box(self._obs_low, self._obs_high) for _ in range(self.n_agents)])

        self._shared_obs_high = np.ones((2 + mask_size) * self.n_agents, dtype=np.float32)
        self._shared_obs_low = np.zeros((2 + mask_size) * self.n_agents, dtype=np.float32)
        self.shared_observation_space = MultiAgentObservationSpace(
            [spaces.Box(self._shared_obs_low, self._shared_obs_high)
                for _ in range(self.n_agents)])

        self.seed()

    def __draw_base_img(self):
        self._base_img = draw_grid(self._grid_shape[0], self._grid_shape[1], cell_size=CELL_SIZE, fill='white')

    def __create_grid(self):
        _grid = [[PRE_IDS['empty'] for _ in range(self._grid_shape[1])] for row in range(self._grid_shape[0])]
        return _grid

    def __update_agent_view(self, agent_i):
        self._full_obs[self.agent_pos[agent_i][0]][self.agent_pos[agent_i][1]] = PRE_IDS['agent'] + str(agent_i + 1)

    def _is_cell_vacant(self, pos):
        return self.is_valid(pos) and (self._full_obs[pos[0]][pos[1]] == PRE_IDS['empty'])

    def __init_full_obs(self, init_pos=None):
        self._full_obs = self.__create_grid()

        for agent_i in range(self.n_agents):
            if init_pos is not None:
                pos = list(init_pos[agent_i])
            else:
                while True:
                    # pos = [self.np_random.randint(0, self._grid_shape[0] - 1),
                    #         self.np_random.randint(0, self._grid_shape[1] - 1)]
                    pos = [
                        self.np_random.randint(
                            self._grid_shape[0] // 3, 
                            self._grid_shape[0] - (self._grid_shape[0] // 3)),
                        self.np_random.randint(
                            self._grid_shape[1] // 3, 
                            self._grid_shape[1] - (self._grid_shape[0] // 3))]
                    if self._is_cell_vacant(pos):
                        # self.agent_pos[agent_i] = pos
                        break
            self.agent_pos[agent_i] = pos
            self.__update_agent_view(agent_i)

        self.__draw_base_img()

    def get_agent_obs(self):
        _obs = []
        for agent_i in range(self.n_agents):
            pos = self.agent_pos[agent_i]
            _agent_i_obs = [pos[0] / (self._grid_shape[0] - 1), pos[1] / (self._grid_shape[1] - 1)]  # coordinates

            _obs_map = np.zeros(self._agent_view_mask)
            obs_range = self._agent_view_mask[0] // 2
            for row in range(max(0, pos[0] - obs_range), min(pos[0] + obs_range + 1, self._grid_shape[0])):
                for col in range(max(0, pos[1] - obs_range), min(pos[1] + obs_range + 1, self._grid_shape[1])):
                    if self._full_obs[row][col] != PRE_IDS["empty"]:
                        if self._see_agents and PRE_IDS['agent'] in self._full_obs[row][col] and self._full_obs[row][col][-1] != str(agent_i + 1):
                            _obs_map[row - (pos[0] - obs_range), col - (pos[1] - obs_range)] = 2

            _agent_i_obs += _obs_map.flatten().tolist()
            _obs.append(_agent_i_obs)

        return _obs

    def reset(self, init_pos=None):
        self._total_episode_reward = [0 for _ in range(self.n_agents)]
        self.agent_pos = {}

        self.__init_full_obs(init_pos)
        self._step_count = 0
        self._steps_beyond_done = None
        self._agent_dones = [False for _ in range(self.n_agents)]

        return self.get_agent_obs()

    def is_valid(self, pos):
        return (0 <= pos[0] < self._grid_shape[0]) and (0 <= pos[1] < self._grid_shape[1])

    # def _is_cell_vacant(self, pos):
    #     return self.is_valid(pos) and (self._full_obs[pos[0]][pos[1]] == PRE_IDS['empty'])

    def __update_agent_pos(self, agent_i, move):

        curr_pos = copy.copy(self.agent_pos[agent_i])
        next_pos = None
        if move == 0:  # down
            next_pos = [curr_pos[0] + 1, curr_pos[1]]
        elif move == 1:  # left
            next_pos = [curr_pos[0], curr_pos[1] - 1]
        elif move == 2:  # up
            next_pos = [curr_pos[0] - 1, curr_pos[1]]
        elif move == 3:  # right
            next_pos = [curr_pos[0], curr_pos[1] + 1]
        elif move == 4:  # no-op
            pass
        else:
            raise Exception('Action Not found!')

        if next_pos is not None and self.is_valid(next_pos):
            self.agent_pos[agent_i] = next_pos
            self._full_obs[curr_pos[0]][curr_pos[1]] = PRE_IDS['empty']
            self.__update_agent_view(agent_i)

    def __next_pos(self, curr_pos, move):
        if move == 0:  # down
            next_pos = [curr_pos[0] + 1, curr_pos[1]]
        elif move == 1:  # left
            next_pos = [curr_pos[0], curr_pos[1] - 1]
        elif move == 2:  # up
            next_pos = [curr_pos[0] - 1, curr_pos[1]]
        elif move == 3:  # right
            next_pos = [curr_pos[0], curr_pos[1] + 1]
        elif move == 4:  # no-op
            next_pos = curr_pos
        return next_pos

    def step(self, agents_action):
        assert (self._step_count is not None), \
            "Call reset before using step method."

        self._step_count += 1
        rewards = [0.0 for _ in range(self.n_agents)]

        for agent_i, action in enumerate(agents_action):
            if not (self._agent_dones[agent_i]):
                self.__update_agent_pos(agent_i, action)

        if (self._step_count >= self._max_steps):
            for i in range(self.n_agents):
                self._agent_dones[i] = True

        for i in range(self.n_agents):
            self._total_episode_reward[i] += rewards[i]

        # Check for episode overflow
        if all(self._agent_dones):
            if self._steps_beyond_done is None:
                self._steps_beyond_done = 0
            else:
                if self._steps_beyond_done == 0:
                    logger.warn(
                        "You are calling 'step()' even though this "
                        "environment has already returned all(done) = True. You "
                        "should always call 'reset()' once you receive "
                        "'all(done) = True' -- any further steps are undefined "
                        "behavior."
                    )
                self._steps_beyond_done += 1

        return self.get_agent_obs(), rewards, self._agent_dones, {}

    def __get_neighbour_coordinates(self, pos):
        neighbours = []
        if self.is_valid([pos[0] + 1, pos[1]]):
            neighbours.append([pos[0] + 1, pos[1]])
        if self.is_valid([pos[0] - 1, pos[1]]):
            neighbours.append([pos[0] - 1, pos[1]])
        if self.is_valid([pos[0], pos[1] + 1]):
            neighbours.append([pos[0], pos[1] + 1])
        if self.is_valid([pos[0], pos[1] - 1]):
            neighbours.append([pos[0], pos[1] - 1])
        return neighbours

    def render(self, mode='human'):
        assert (self._step_count is not None), \
            "Call reset before using render method."
        # print(self._full_obs)
        img = copy.copy(self._base_img)
        for agent_i in range(self.n_agents):
            for neighbour in self.__get_neighbour_coordinates(self.agent_pos[agent_i]):
                fill_cell(img, neighbour, cell_size=CELL_SIZE, fill=AGENT_NEIGHBORHOOD_COLOR, margin=0.1)
            fill_cell(img, self.agent_pos[agent_i], cell_size=CELL_SIZE, fill=AGENT_NEIGHBORHOOD_COLOR, margin=0.1)

        for agent_i in range(self.n_agents):
            draw_circle(img, self.agent_pos[agent_i], cell_size=CELL_SIZE, fill=AGENT_COLOR)
            write_cell_text(img, text=str(agent_i + 1), pos=self.agent_pos[agent_i], cell_size=CELL_SIZE,
                            fill='white', margin=0.4)

        img = np.asarray(img)
        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img)
            return self.viewer.isopen

    def seed(self, n=None):
        self.np_random, seed = seeding.np_random(n)
        return [seed]

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


AGENT_COLOR = ImageColor.getcolor('blue', mode='RGB')
AGENT_NEIGHBORHOOD_COLOR = (186, 238, 247)

CELL_SIZE = 35

WALL_COLOR = 'black'

ACTION_MEANING = {
    0: "DOWN",
    1: "LEFT",
    2: "UP",
    3: "RIGHT",
    4: "NOOP",
}

PRE_IDS = {
    'agent': 'A',
    'wall': 'W',
    'empty': '0'
}