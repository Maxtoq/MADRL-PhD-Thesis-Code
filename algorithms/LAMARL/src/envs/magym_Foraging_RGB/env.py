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


class Env(gym.Env):
    """
    
    """
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, grid_shape=(15, 15), n_agents=4, n_gems=30, penalty=-0.0, 
                 step_cost=-1.0, max_steps=100, obs_range=5, reduced_obsrange=None, 
                 respawn_gems=False):
        assert len(grid_shape) == 2, 'expected a tuple of size 2 for grid_shape, but found {}'.format(grid_shape)
        assert grid_shape[0] > 0 and grid_shape[1] > 0, 'grid shape should be > 0'
        assert 0 < obs_range <= grid_shape[0], 'obs_range has to be within (0,{}]'.format(grid_shape[0])
        assert n_agents >= 3, "n_agents must be > 2."
        if reduced_obsrange is not None:
            assert reduced_obsrange < obs_range, "reduced_obsrange should not be larger than obs_range"

        self._grid_shape = grid_shape
        self.n_agents = n_agents
        self.n_gems = n_gems
        self._max_steps = max_steps
        self._step_count = 0
        self._penalty = penalty
        self._step_cost = step_cost
        self._agent_view_mask = (obs_range, obs_range, 3) # 3 canals for Red, GReen, and Green
        self._reduced_obsrange = reduced_obsrange
        self._respawn_gems = respawn_gems

        self._agent_init_pos = [
            [grid_shape[0] // 2 - 1, grid_shape[1] // 2 - 1],
            [grid_shape[0] // 2 - 1, grid_shape[1] // 2],
            [grid_shape[0] // 2, grid_shape[1] // 2 - 1],
            [grid_shape[0] // 2, grid_shape[1] // 2]]
        self._gem_forbid_pos = [
            [0, 0],
            [0, grid_shape[1] - 1],
            [grid_shape[0] - 1, 0],
            [grid_shape[0] - 1, grid_shape[1] - 1]]

        self.action_space = MultiAgentActionSpace([spaces.Discrete(5) for _ in range(self.n_agents)])
        self.agent_pos = {_: None for _ in range(self.n_agents)}

        # Init gem set
        self.gem_pos = {_: None for _ in range(self.n_gems)}
        self.gem_colors = [3, 3, 3, 2, 2, 2, 2, 2, 2, 2]
        self.gem_colors += [1] * (self.n_gems - len(self.gem_colors))
        # for g_i in range(len(self.gem_colors), self.n_gems):
        #     self.gem_colors.append(1)
        self._gem_alive = None

        self._base_grid = self.__create_grid()  # with no agents
        self._full_obs = self.__create_grid()

        self._agent_dones = [False for _ in range(self.n_agents)]
        self.viewer = None

        # agent pos (2), gem (25), step (1)
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

        self._total_episode_reward = None
        self.seed()

    def get_action_meanings(self, agent_i=None):
        if agent_i is not None:
            assert agent_i <= self.n_agents
            return [ACTION_MEANING[i] for i in range(self.action_space[agent_i].n)]
        else:
            return [[ACTION_MEANING[i] for i in range(ac.n)] for ac in self.action_space]

    def action_space_sample(self):
        return [agent_action_space.sample() for agent_action_space in self.action_space]

    def __draw_base_img(self):
        self._base_img = draw_grid(self._grid_shape[0], self._grid_shape[1], cell_size=CELL_SIZE, fill='white')

    def __create_grid(self):
        _grid = [[ENT_IDS['empty'] for _ in range(self._grid_shape[1])] for row in range(self._grid_shape[0])]
        return _grid

    def __spawn_gem(self, gem_i):
        while True:
            pos = [self.np_random.randint(0, self._grid_shape[0]),
                    self.np_random.randint(0, self._grid_shape[1])]
            # print(self._is_cell_vacant(pos), (self._neighbour_agents(pos)[0] == 0), (pos not in self._gem_forbid_pos))
            if self._is_cell_vacant(pos) and (self._neighbour_agents(pos)[0] == 0) and (pos not in self._gem_forbid_pos):
                self.gem_pos[gem_i] = pos
                break

        self.__update_gem_view(gem_i)

    def __init_full_obs(self):
        self._full_obs = self.__create_grid()

        for agent_i in range(self.n_agents):
            # while True:
            #     pos = [self.np_random.randint(0, self._grid_shape[0] - 1),
            #            self.np_random.randint(0, self._grid_shape[1] - 1)]
            #     if self._is_cell_vacant(pos):
            #         self.agent_pos[agent_i] = pos
                    # break
            self.agent_pos[agent_i] = self._agent_init_pos[agent_i]
            self.__update_agent_view(agent_i)

        for gem_i in range(self.n_gems):
            self.__spawn_gem(gem_i)
        self.__draw_base_img()

    def get_agent_obs(self):
        _obs = []
        for agent_i in range(self.n_agents):
            pos = self.agent_pos[agent_i]
            _agent_i_obs = [pos[0] / (self._grid_shape[0] - 1), pos[1] / (self._grid_shape[1] - 1)]  # coordinates

            # check if gem is in the view area
            _gem_pos = np.zeros(self._agent_view_mask)  # gem location in neighbour
            obs_range = self._agent_view_mask[0] // 2
            for row in range(max(0, pos[0] - obs_range), min(pos[0] + obs_range + 1, self._grid_shape[0])):
                for col in range(max(0, pos[1] - obs_range), min(pos[1] + obs_range + 1, self._grid_shape[1])):
                    if self._full_obs[row][col] != ENT_IDS["empty"]:
                        # If limited obs_range, then check if distance is low enough
                        if self._reduced_obsrange is not None:
                            dist = np.sqrt((row - pos[0]) ** 2 + (col - pos[1]) ** 2)
                            if dist > self._reduced_obsrange / 2:
                                continue
                        if ENT_IDS['agent'] in self._full_obs[row][col]: # and self._full_obs[row][col][-1] != str(agent_i + 1):
                            _gem_pos[row - (pos[0] - obs_range), col - (pos[1] - obs_range), 2] = 1 # Agent is blue, so observe (0, 0, 1)
                        elif ENT_IDS['gem'] in self._full_obs[row][col]:
                            _gem_pos[row - (pos[0] - obs_range), col - (pos[1] - obs_range)] = GEM_COLORS[int(self._full_obs[row][col][-1])]
            
            _agent_i_obs += _gem_pos.flatten().tolist()  # adding gem pos in observable area
            _obs.append(_agent_i_obs)
        return _obs

    def reset(self, init_pos=None):
        self._total_episode_reward = [0 for _ in range(self.n_agents)]
        self.agent_pos = {}
        self.gem_pos = {}

        # TODO handle init_pos
        self.__init_full_obs()
        self._step_count = 0
        self._agent_dones = [False for _ in range(self.n_agents)]
        self._gem_alive = [True for _ in range(self.n_gems)]

        return self.get_agent_obs()

    def __wall_exists(self, pos):
        row, col = pos
        return ENT_IDS['wall'] in self._base_grid[row, col]

    def is_valid(self, pos):
        return (0 <= pos[0] < self._grid_shape[0]) and (0 <= pos[1] < self._grid_shape[1])

    def _is_cell_vacant(self, pos):
        return self.is_valid(pos) and (self._full_obs[pos[0]][pos[1]] == ENT_IDS['empty'])

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

        if next_pos is not None and self._is_cell_vacant(next_pos):
            self.agent_pos[agent_i] = next_pos
            self._full_obs[curr_pos[0]][curr_pos[1]] = ENT_IDS['empty']
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

    def __update_agent_view(self, agent_i):
        self._full_obs[self.agent_pos[agent_i][0]][self.agent_pos[agent_i][1]] = ENT_IDS['agent'] + str(agent_i + 1)

    def __update_gem_view(self, gem_i):
        self._full_obs[self.gem_pos[gem_i][0]][self.gem_pos[gem_i][1]] = ENT_IDS['gem'] + str(self.gem_colors[gem_i])#str(gem_i + 1)

    def _neighbour_agents(self, pos):
        # check if agent is in neighbour
        _count = 0
        neighbours_xy = []
        if self.is_valid([pos[0] + 1, pos[1]]) and ENT_IDS['agent'] in self._full_obs[pos[0] + 1][pos[1]]:
            _count += 1
            neighbours_xy.append([pos[0] + 1, pos[1]])
        if self.is_valid([pos[0] - 1, pos[1]]) and ENT_IDS['agent'] in self._full_obs[pos[0] - 1][pos[1]]:
            _count += 1
            neighbours_xy.append([pos[0] - 1, pos[1]])
        if self.is_valid([pos[0], pos[1] + 1]) and ENT_IDS['agent'] in self._full_obs[pos[0]][pos[1] + 1]:
            _count += 1
            neighbours_xy.append([pos[0], pos[1] + 1])
        if self.is_valid([pos[0], pos[1] - 1]) and ENT_IDS['agent'] in self._full_obs[pos[0]][pos[1] - 1]:
            neighbours_xy.append([pos[0], pos[1] - 1])
            _count += 1

        agent_id = []
        for x, y in neighbours_xy:
            agent_id.append(int(self._full_obs[x][y].split(ENT_IDS['agent'])[1]) - 1)
        return _count, agent_id

    def step(self, agents_action):
        self._step_count += 1
        rewards = [self._step_cost for _ in range(self.n_agents)]

        for agent_i, action in enumerate(agents_action):
            if not (self._agent_dones[agent_i]):
                self.__update_agent_pos(agent_i, action)

        for gem_i in range(self.n_gems):
            if self._gem_alive[gem_i]:
                predator_neighbour_count, n_i = self._neighbour_agents(self.gem_pos[gem_i])

                if predator_neighbour_count > 0:
                    if predator_neighbour_count >= self.gem_colors[gem_i]:
                        _reward = GEM_REWARDS[self.gem_colors[gem_i]]
                        self._gem_alive[gem_i] = False
                        self._full_obs[self.gem_pos[gem_i][0]][self.gem_pos[gem_i][1]] = ENT_IDS['empty']
                    else:
                        _reward = self._penalty

                    for agent_i in range(self.n_agents):
                        rewards[agent_i] += _reward
            # Respawn
            elif self._respawn_gems:
                prob_respawn = 1 / GEM_REWARDS[self.gem_colors[gem_i]]
                if np.random.random() < prob_respawn:
                    self.__spawn_gem(gem_i)
                    self._gem_alive[gem_i] = True


        if (self._step_count >= self._max_steps) or (True not in self._gem_alive):
            for i in range(self.n_agents):
                self._agent_dones[i] = True

        for i in range(self.n_agents):
            self._total_episode_reward[i] += rewards[i]

        return self.get_agent_obs(), rewards, self._agent_dones, {'gem_alive': self._gem_alive}

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
        img = copy.copy(self._base_img)
        for agent_i in range(self.n_agents):
            for neighbour in self.__get_neighbour_coordinates(self.agent_pos[agent_i]):
                fill_cell(img, neighbour, cell_size=CELL_SIZE, fill=AGENT_NEIGHBORHOOD_COLOR, margin=0.1)
            fill_cell(img, self.agent_pos[agent_i], cell_size=CELL_SIZE, fill=AGENT_NEIGHBORHOOD_COLOR, margin=0.1)

        for agent_i in range(self.n_agents):
            draw_circle(img, self.agent_pos[agent_i], cell_size=CELL_SIZE, fill=AGENT_COLOR)
            write_cell_text(img, text=str(agent_i + 1), pos=self.agent_pos[agent_i], cell_size=CELL_SIZE,
                            fill='white', margin=0.4)

        for gem_i in range(self.n_gems):
            if self._gem_alive[gem_i]:
                draw_circle(img, self.gem_pos[gem_i], cell_size=CELL_SIZE, fill=GEM_RENDER_COLORS[self.gem_colors[gem_i]])
                write_cell_text(img, text=str(gem_i + 1), pos=self.gem_pos[gem_i], cell_size=CELL_SIZE,
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

GEM_RENDER_COLORS = {
    1: 'yellow',
    2: 'green',
    3: 'purple'
}
GEM_REWARDS = {
    1: 1,
    2: 5,
    3: 20
}

GEM_COLORS = {
    1: [1, 1, 0], # yellow
    2: [0, 1, 1], # green
    3: [1, 0, 1]  # purple
}

CELL_SIZE = 35

WALL_COLOR = 'black'

ACTION_MEANING = {
    0: "DOWN",
    1: "LEFT",
    2: "UP",
    3: "RIGHT",
    4: "NOOP",
}

ENT_IDS = {
    'agent': 'A',
    'gem': 'G',
    'wall': 'W',
    'empty': '0'
}
