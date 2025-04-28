import random
import numpy as np

from ..mpe.core import Agent, World, Landmark
from ..mpe.scenario import BaseScenario


AGENT_RADIUS = 0.05
AGENT_MASS = 0.5

OBS_RANGE = 0.5

LM_COLORS = {
    "Red": np.array([0.75, 0.25, 0.25]),
    "Light Red": np.array([0.9, 0.5, 0.5]),
    "Dark Red": np.array([0.6, 0.05, 0.05]),
    "Green": np.array([0.25, 0.75, 0.25]),
    "Light Green": np.array([0.5, 0.9, 0.5]),
    "Dark Green": np.array([0.05, 0.6, 0.05]),
    "Blue": np.array([0.25, 0.25, 0.75]),
    "Light Blue": np.array([0.5, 0.5, 0.9]),
    "Dark Blue": np.array([0.05, 0.05, 0.6]),
}


def get_dist(pos1, pos2, squared=False):
    dist = np.sum(np.square(pos1 - pos2))
    if squared:
        return dist
    else:
        return np.sqrt(dist)

    
class Scenario(BaseScenario):

    def make_world(self, max_steps=100, n_landmarks=5, env_size=5):
        self.max_steps = max_steps
        self.n_landmarks = n_landmarks
        self.env_size = env_size

        self.world = World()
        # set any world properties first
        self.world.dim_c = 0 # No communication via mpe
        self.world.collaborative = True  # whether agents share rewards
        # add agents
        self.world.agents = [Agent() for i in range(2)]
        for i, agent in enumerate(self.world.agents):
            agent.name = f"agent_{i}"
            agent.collide = False
            agent.size = AGENT_RADIUS
            agent.initial_mass = AGENT_MASS
            agent.silent = True
        # add landmarks
        self.world.landmarks = [Landmark() for i in range(self.n_landmarks)]
        for i, landmark in enumerate(self.world.landmarks):
            landmark.name = "landmark %d" % i
            landmark.collide = False
            landmark.movable = False

        # make initial conditions
        self.reset_world()

        return self.world

    def reset_world(self, seed=None):
        # assign goals to agents
        for agent in self.world.agents:
            agent.goal_a = None
            agent.goal_lm = None
        # want other agent to go to the goal landmark
        self.world.agents[0].goal_a = self.world.agents[1]
        self.world.agents[0].goal_lm = np.random.choice(self.world.landmarks)
        self.world.agents[1].goal_a = self.world.agents[0]
        self.world.agents[1].goal_lm = np.random.choice(self.world.landmarks)
        # random properties for agents
        for i, agent in enumerate(self.world.agents):
            agent.color = np.array([0.25, 0.25, 0.25])
        # random properties for landmarks
        # self.world.landmarks[0].color = colors[0]
        # self.world.landmarks[1].color = colors[1]
        # self.world.landmarks[2].color = colors[2]
        # special colors for goals
        self.world.agents[0].goal_a.color = self.world.agents[0].goal_lm.color
        self.world.agents[1].goal_a.color = self.world.agents[1].goal_lm.color
        # set random initial states
        for agent in self.world.agents:
            agent.state.p_pos = np.random.uniform(-self.env_size, +self.env_size, self.world.dim_p)
            agent.state.p_vel = np.zeros(self.world.dim_p)
            agent.state.c = np.zeros(self.world.dim_c)

        colors = random.sample(list(LM_COLORS.values()), len(LM_COLORS))
        for i, landmark in enumerate(self.world.landmarks):
            landmark.state.p_pos = np.random.uniform(-self.env_size, +self.env_size, self.world.dim_p)
            landmark.state.p_vel = np.zeros(self.world.dim_p)

            landmark.color = colors[i]

        self.world.current_step = 0

    def done(self, agent):
        # Done if all preys are caught
        return self.world.current_step >= self.max_steps

    def reward(self, agent):
        if agent.goal_a is None or agent.goal_lm is None:
            agent_reward = 0.0
        else:
            agent_reward = np.sqrt(
                np.sum(np.square(agent.goal_a.state.p_pos - agent.goal_lm.state.p_pos))
            )
        return -agent_reward

    def global_reward(self):
        all_rewards = sum(self.reward(agent, self.world) for agent in self.world.agents)
        return all_rewards / len(self.world.agents)

    def observation(self, agent):
        """
        Obs:
        - Agent pos (dim: 2)
        - Agent vel (dim: 2)
        - for each landmark: [seen (1 or 0), rel_pos, color] (dim: 6)
        - goal: [pos, color] (dim: 5)
        """
        obs = [agent.state.p_pos, agent.state.p_vel]

        for l in self.world.landmarks:
            if get_dist(agent.state.p_pos, l.state.p_pos) <= OBS_RANGE:
                obs.append(np.concatenate((
                    [1.0],
                    (l.state.p_pos - agent.state.p_pos) / OBS_RANGE, # Relative position normailised into [0, 1]
                    l.color # Velocity
                )))
            else:
                obs.append(np.array([0.0, 1.0, 1.0, 0.0, 0.0, 0.0]))

        obs.append(agent.goal_lm.state.p_pos)
        obs.append(agent.goal_lm.color)

        obs.append([self.world.current_step / self.max_steps])

        return np.concatenate(obs)


