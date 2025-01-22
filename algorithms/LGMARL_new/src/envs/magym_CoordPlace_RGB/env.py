import copy
import logging
import random
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
    Coordinated placement: 
        Agents have to navigate, find landmarks and 
        communicate the color to their partner so it goes to a landmark of the 
        same color.
    Observations: 
        Agents observe only their position, the color of the
        landmark they are on (if on a landmark), and the current step of the 
        episode.
    Rewards: 
        Penalty of -1 at each time step, episode stops if all agents are 
        landmarks with same color.
    """
    metadata = {'render.modes': ['human', 'rgb_array']}
    _landmark_sets = {2: (2, 2, 1), 3: (3, 2)}

    def __init__(self, n_agents=2, step_cost=-1.0, max_steps=50):
        assert n_agents in self._landmark_sets, f"Bad number of agents, must be in {list(self._landmark_sets.keys())}."
        self._grid_shape = (9, 9)
        self.n_agents = n_agents
        self._max_steps = max_steps
        self._step_count = 0
        self._step_cost = step_cost
        self.n_landmarks = 5

        self.action_space = MultiAgentActionSpace(
            [spaces.Discrete(5) for _ in range(self.n_agents)])

        self.agent_pos = {_: None for _ in range(self.n_agents)}

        self.lm_sets = self._landmark_sets[self.n_agents]
        self.lm_positions = [None for _ in range(self.n_landmarks)]
        self.lm_colors = [None for _ in range(self.n_landmarks)]

        self._base_grid = self.__create_grid()  # with no agents
        self._full_obs = self.__create_grid()

        self._agent_dones = [False for _ in range(self.n_agents)]
        self.viewer = None

        # obs = pos (2) + color (3) + step (1)
        self._obs_high = np.ones(2 + 3 + 1, dtype=np.float32)
        self._obs_low = np.zeros(2 + 3 + 1, dtype=np.float32)
        self.observation_space = MultiAgentObservationSpace(
            [spaces.Box(self._obs_low, self._obs_high) 
             for _ in range(self.n_agents)])

        self._shared_obs_high = np.ones(
            (2 + 3 + 1) * self.n_agents, dtype=np.float32)
        self._shared_obs_low = np.zeros(
            (2 + 3 + 1) * self.n_agents, dtype=np.float32)
        self.shared_observation_space = MultiAgentObservationSpace(
            [spaces.Box(self._shared_obs_low, self._shared_obs_high)
                for _ in range(self.n_agents)])

        self._total_episode_reward = None
        self.seed()

    def get_action_meanings(self, agent_i=None):
        if agent_i is not None:
            assert agent_i <= self.n_agents
            return [ACTION_MEANING[i] 
                    for i in range(self.action_space[agent_i].n)]
        else:
            return [[ACTION_MEANING[i] 
                     for i in range(ac.n)] for ac in self.action_space]

    def action_space_sample(self):
        return [agent_action_space.sample() for agent_action_space in self.action_space]

    def __draw_base_img(self):
        self._base_img = draw_grid(self._grid_shape[0], self._grid_shape[1], cell_size=CELL_SIZE, fill='white')

    def __create_grid(self):
        _grid = [[ENT_IDS['empty'] for _ in range(self._grid_shape[1])] for row in range(self._grid_shape[0])]
        return _grid
    
    def __put_landmarks(self):
        for lp, lc in zip(self.lm_positions, self.lm_colors):
            r, c = lp
            cell = ENT_IDS['landmark'] + str(LM_COLORS[lc])
            self._full_obs[r][c] = cell
            self._full_obs[r + 1][c] = cell
            self._full_obs[r + 2][c] = cell
            self._full_obs[r][c + 1] = cell
            self._full_obs[r + 1][c + 1] = cell
            self._full_obs[r + 2][c + 1] = cell
            self._full_obs[r][c + 2] = cell
            self._full_obs[r + 1][c + 2] = cell
            self._full_obs[r + 2][c + 2] = cell

    def __put_agent_i(self, a_i):
        self._full_obs[self.agent_pos[a_i][0]][self.agent_pos[a_i][1]] = ENT_IDS['agent'] + str(a_i + 1)

    def __make_full_obs(self):
        self.__put_landmarks()
        for a_i in range(self.n_agents):
            self.__put_agent_i(a_i)

    def __init_landmarks(self):
        available_colors = list(LM_COLORS.keys())
        available_positions = list(range(8))

        lm_i = 0
        for n in self.lm_sets:
            color = random.choice(available_colors)
            available_colors.remove(color)

            for _ in range(n):
                p_i = random.choice(available_positions)
                available_positions.remove(p_i)

                self.lm_positions[lm_i] = LM_POSITIONS[p_i]
                self.lm_colors[lm_i] = color
                lm_i += 1

    def __init_full_obs(self):
        self._full_obs = self.__create_grid()

        for a_i in range(self.n_agents):
            while True:
                pos = [self.np_random.randint(3, 6), self.np_random.randint(3, 6)]
                if self.__is_cell_vacant(pos):
                    self.agent_pos[a_i] = pos
                    break
            self.__put_agent_i(a_i)

        self.__init_landmarks()
        self.__make_full_obs()
        self.__draw_base_img()

    def __pos_on_lm(self, lm_pos, pos):
        return (lm_pos[0] <= pos[0] <= lm_pos[0] + 2) \
            and (lm_pos[1] <= pos[1] <= lm_pos[1] + 2)

    def __get_position_lm(self, pos):
        for l_i, l_p in enumerate(self.lm_positions):
            if self.__pos_on_lm(l_p, pos):
                return l_i, self.lm_colors[l_i]
        return -1, -1 # for "empty"

    def get_agent_obs(self):
        _obs = []
        for a_i in range(self.n_agents):
            pos = self.agent_pos[a_i]
            _agent_i_obs = [pos[0] / (self._grid_shape[0] - 1), pos[1] / (self._grid_shape[1] - 1)]  # coordinates

            # TODO: add color of position
            color = self.__get_position_lm(pos)[1]
            if color == -1:
                _agent_i_obs += [0, 0, 0]
            else:
                _agent_i_obs += LM_COLORS[color]

            _agent_i_obs.append(self._step_count / self._max_steps)
            
            _obs.append(_agent_i_obs)
        return _obs

    def reset(self, init_pos=None):
        self._total_episode_reward = [0 for _ in range(self.n_agents)]
        self.agent_pos = {}

        self.__init_full_obs()
        self._step_count = 0
        self._agent_dones = [False for _ in range(self.n_agents)]

        return self.get_agent_obs()

    def __is_valid(self, pos):
        return (0 <= pos[0] < self._grid_shape[0]) and (0 <= pos[1] < self._grid_shape[1])

    def __is_cell_vacant(self, pos):
        return self.__is_valid(pos) and (not self._full_obs[pos[0]][pos[1]].startswith(ENT_IDS['agent']))

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

        if next_pos is not None and self.__is_cell_vacant(next_pos):
            self.agent_pos[agent_i] = next_pos
            self._full_obs[curr_pos[0]][curr_pos[1]] = ENT_IDS['empty']
            self.__put_agent_i(agent_i)

    def step(self, agents_action):
        self._step_count += 1
        rewards = [self._step_cost for _ in range(self.n_agents)]

        for a_i, action in enumerate(agents_action):
            if not (self._agent_dones[a_i]):
                self.__update_agent_pos(a_i, action)

        # Redraw to have landmarks
        self.__make_full_obs()

        # Find if agents are on different landmarks with same colors
        agent_colors = []
        agent_lms = []
        for a_i in range(self.n_agents):
            i, c = self.__get_position_lm(self.agent_pos[a_i])
            agent_colors.append(c)
            agent_lms.append(i)
        all_same_color = len(set(agent_colors)) <= 1
        all_diff_landmark = len(set(agent_lms)) == len(agent_lms)
        done = all_same_color and all_diff_landmark

        if (self._step_count >= self._max_steps) or done:
            for i in range(self.n_agents):
                self._agent_dones[i] = True

        for i in range(self.n_agents):
            self._total_episode_reward[i] += rewards[i]

        return self.get_agent_obs(), rewards, self._agent_dones, {}

    def render(self, mode='human'):
        img = copy.copy(self._base_img)
        for l_i, l_p in enumerate(self.lm_positions):
            r, c = l_p
            fill_cell(img, [r, c], cell_size=CELL_SIZE, fill=LM_RENDER_COLORS[self.lm_colors[l_i]], margin=0.1)
            fill_cell(img, [r + 1, c], cell_size=CELL_SIZE, fill=LM_RENDER_COLORS[self.lm_colors[l_i]], margin=0.1)
            fill_cell(img, [r + 2, c], cell_size=CELL_SIZE, fill=LM_RENDER_COLORS[self.lm_colors[l_i]], margin=0.1)
            fill_cell(img, [r, c + 1], cell_size=CELL_SIZE, fill=LM_RENDER_COLORS[self.lm_colors[l_i]], margin=0.1)
            fill_cell(img, [r + 1, c + 1], cell_size=CELL_SIZE, fill=LM_RENDER_COLORS[self.lm_colors[l_i]], margin=0.1)
            fill_cell(img, [r + 2, c + 1], cell_size=CELL_SIZE, fill=LM_RENDER_COLORS[self.lm_colors[l_i]], margin=0.1)
            fill_cell(img, [r, c + 2], cell_size=CELL_SIZE, fill=LM_RENDER_COLORS[self.lm_colors[l_i]], margin=0.1)
            fill_cell(img, [r + 1, c + 2], cell_size=CELL_SIZE, fill=LM_RENDER_COLORS[self.lm_colors[l_i]], margin=0.1)
            fill_cell(img, [r + 2, c + 2], cell_size=CELL_SIZE, fill=LM_RENDER_COLORS[self.lm_colors[l_i]], margin=0.1)

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

LM_RENDER_COLORS = {
    1: 'yellow',
    2: 'cyan',
    3: 'purple',
    4: 'red',
    5: 'green',
    6: 'blue'
}

LM_COLORS = {
    1: [1, 1, 0], # yellow
    2: [0, 1, 1], # cyan
    3: [1, 0, 1], # purple
    4: [1, 0, 0], # red
    5: [0, 1, 0], # green
    6: [0, 0, 1]  # blue
}

LM_POSITIONS = [
    (0, 0),
    (3, 0),
    (6, 0),
    (0, 3),
    (6, 3),
    (0, 6),
    (3, 6),
    (6, 6)
]

CELL_SIZE = 35

ACTION_MEANING = {
    0: "DOWN",
    1: "LEFT",
    2: "UP",
    3: "RIGHT",
    4: "NOOP",
}

ENT_IDS = {
    'agent': 'A',
    'landmark': 'L',
    'empty': '0'
}
