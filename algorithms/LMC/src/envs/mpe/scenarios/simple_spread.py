import numpy as np
from ..core import World, Agent, Landmark
from ..scenario import BaseScenario


class Scenario(BaseScenario):
    def make_world(self):
        self.world = World()
        # set any world properties first
        self.world.dim_c = 2
        num_agents = 3
        num_landmarks = 3
        self.world.collaborative = True
        # add agents
        self.world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(self.world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.size = 0.15
        # add landmarks
        self.world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(self.world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
        # make initial conditions
        self.reset_world()

    def seed(self, seed):
        np.random.seed(seed)
        random.seed(seed)

    def reset_world(self):
        # random properties for agents
        for i, agent in enumerate(self.world.agents):
            agent.color = np.array([0.35, 0.35, 0.85])
        # random properties for landmarks
        for i, landmark in enumerate(self.world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
        # set random initial states
        for agent in self.world.agents:
            agent.state.p_pos = np.random.uniform(-1, +1, self.world.dim_p)
            agent.state.p_vel = np.zeros(self.world.dim_p)
            agent.state.c = np.zeros(self.world.dim_c)
        for i, landmark in enumerate(self.world.landmarks):
            landmark.state.p_pos = np.random.uniform(-1, +1, self.world.dim_p)
            landmark.state.p_vel = np.zeros(self.world.dim_p)

    def benchmark_data(self, agent):
        rew = 0
        collisions = 0
        occupied_landmarks = 0
        min_dists = 0
        for l in self.world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) 
                        for a in self.world.agents]
            min_dists += min(dists)
            rew -= min(dists)
            if min(dists) < 0.1:
                occupied_landmarks += 1
        if agent.collide:
            for a in self.world.agents:
                if self.is_collision(a, agent):
                    rew -= 1
                    collisions += 1
        return (rew, collisions, min_dists, occupied_landmarks)


    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def reward(self, agent):
        # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
        rew = 0
        for l in self.world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) 
                        for a in self.world.agents]
            rew -= min(dists)
        if agent.collide:
            for a in self.world.agents:
                if self.is_collision(a, agent):
                    rew -= 1
        return rew

    def observation(self, agent):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in self.world.landmarks:  # world.entities:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # entity colors
        entity_color = []
        for entity in self.world.landmarks:  # world.entities:
            entity_color.append(entity.color)
        # communication of all other agents
        comm = []
        other_pos = []
        for other in self.world.agents:
            if other is agent: continue
            comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)
        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + comm)
