import numpy as np
import random

from multiagent.core import Walled_World, Agent, Landmark
from multiagent.scenario import BaseScenario

LANDMARK_SIZE = 0.06
AGENT_SIZE = 0.06

def get_dist(pos1, pos2, squared=False):
    dist = np.sum(np.square(pos1 - pos2))
    if squared:
        return dist
    else:
        return np.sqrt(dist)


class SimpleSpreadWorld(Walled_World):

    def __init__(self):
        super(SimpleSpreadWorld, self).__init__()
        self.landmarks_filled = []

    def step(self):
        super().step()
        for i, l in enumerate(self.landmarks):
            for a in self.agents:
                if get_dist(a.state.p_pos, l.state.p_pos) < LANDMARK_SIZE:
                    self.landmarks_filled[i] = True
                    break
                else:
                    self.landmarks_filled[i] = False


class Scenario(BaseScenario):
    def make_world(self, **kwargs):
        world = SimpleSpreadWorld()
        # set any world properties first
        world.dim_c = 2
        self.nb_agents = kwargs["nb_agents"]
        num_landmarks = self.nb_agents
        world.collaborative = False
        # add agents
        world.agents = [Agent() for i in range(self.nb_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.size = 0.15
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
            landmark.size = LANDMARK_SIZE
            world.landmarks_filled.append(False)
        # make initial conditions
        self.reset_world(world)
        self.done_flag = False
        return world

    def reset_world(self, world, seed=None, init_pos=None):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.35, 0.85])
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(
                -1 + AGENT_SIZE, 1 - AGENT_SIZE, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        corners = random.sample([
                [[-1 + LANDMARK_SIZE, 0 - LANDMARK_SIZE], 
                 [0 + LANDMARK_SIZE, 1 - LANDMARK_SIZE]],
                [[0 + LANDMARK_SIZE, 1 - LANDMARK_SIZE], 
                 [0 + LANDMARK_SIZE, 1 - LANDMARK_SIZE]],
                [[-1 + LANDMARK_SIZE, 0 - LANDMARK_SIZE], 
                 [-1 + LANDMARK_SIZE, 0 - LANDMARK_SIZE]],
                [[0 + LANDMARK_SIZE, 1 - LANDMARK_SIZE], 
                 [-1 + LANDMARK_SIZE, 0 - LANDMARK_SIZE]]],
            self.nb_agents)
        for i, landmark in enumerate(world.landmarks):
            x = random.uniform(*corners[i][0])
            y = random.uniform(*corners[i][1])
            landmark.state.p_pos = np.array([x, y])
            landmark.state.p_vel = np.zeros(world.dim_p)

    def benchmark_data(self, agent, world):
        rew = 0
        collisions = 0
        occupied_landmarks = 0
        min_dists = 0
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
            min_dists += min(dists)
            rew -= min(dists)
            if min(dists) < 0.1:
                occupied_landmarks += 1
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1
                    collisions += 1
        return (rew, collisions, min_dists, occupied_landmarks)


    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def done(self, agent, world):
        return self.done_flag

    def reward(self, agent, world):
        # agents_on_lms = []
        # # Done if all agents are on a landmark
        # for a in world.agents:
        #     on_lms = [
        #         get_dist(a.state.p_pos, l.state.p_pos) < LANDMARK_SIZE
        #         for l in world.landmarks
        #     ]
        #     agents_on_lms.append(any(on_lms))
        self.done_flag = all(world.landmarks_filled)
        
        rew = 100.0 if self.done_flag else 0.0

        rew += -0.1

        return rew

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        lm_pos = []
        for i, l in enumerate(world.landmarks):  # world.entities:
            lm_pos.append(np.concatenate((
                [int(world.landmarks_filled[i])],
                l.state.p_pos - agent.state.p_pos
            )))
        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + lm_pos)
