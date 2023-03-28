import numpy as np
import random

from multiagent.scenario import BaseScenario
from multiagent.core import Walled_World, Agent, Landmark, Action, Entity

AGENT_RADIUS = 0.04
AGENT_MASS = 0.4

def get_dist(pos1, pos2, squared=False):
    dist = np.sum(np.square(pos1 - pos2))
    if squared:
        return dist
    else:
        return np.sqrt(dist)
        

class Button(Landmark):

    def __init__(self):
        super(Button, self).__init__()
        self.collide = False
        self.color_name = None

    def is_pushing(self, agent_pos):
        return get_dist(agent_pos, self.state.p_pos) < BUTTON_RADIUS

class Chunk(Entity):

    colors = {
        "red": np.array([1.0, 0.0, 0.0]),
        "green": np.array([0.0, 1.0, 0.0]),
        "blue": np.array([0.0, 0.0, 1.0]),
        "yellow": np.array([1.0, 1.0, 0.0]),
        "pink": np.array([0.0, 1.0, 1.0])
    }

    def __init__(self):
        super(Chunk, self).__init__()
        self.movable = False
        self.color = None
        self.size = 0
        self.nb_agents_touched = 0
        self.done = False
        self.pos_id = -1


class ForagingWorld(Walled_World):

    def __init__(self, nb_agents, scenario_params):
        super(ForagingWorld, self).__init__()
        # add agent
        self.nb_agents = nb_agents
        self.agents = [Agent() for i in range(nb_agents)]
        # Resources
        self.chunks = [Chunk() for i in range(2)]
        # Control inertia
        self.damping = 0.8
        self.scenario_params = scenario_params

    @property
    def entities(self):
        return self.agents + self.chunks

    def init_pos_chunks(self, start=False):
        small_chunk_positions = [
            np.array([-0.4, 0.0]),
            np.array([0.4, 0.0]),
            np.array([0.0, -0.4]),
            np.array([0.0, 0.4])]
        big_chunk_positions = [
            np.array([-0.8, 0.8]),
            np.array([0.8, 0.8]),
            np.array([-0.8, -0.8]),
            np.array([0.8, -0.8])]
        if self.chunks[0].done or start:
            if self.chunks[0].pos_id == -1:
                self.chunks[0].pos_id = random.randint(0, 3)
            else:
                while True:
                    new_id = random.randint(0, 3)
                    if new_id != self.chunks[0].pos_id:
                        break
                self.chunks[0].pos_id = new_id
            self.chunks[0].state.p_pos = small_chunk_positions[
                self.chunks[0].pos_id]
            self.chunks[0].done =False
        if self.chunks[1].done or start:
            if self.chunks[1].pos_id == -1:
                self.chunks[1].pos_id = random.randint(0, 3)
            else:
                while True:
                    new_id = random.randint(0, 3)
                    if new_id != self.chunks[1].pos_id:
                        break
                self.chunks[1].pos_id = self.chunks[1].pos_id - 1
            self.chunks[1].state.p_pos = big_chunk_positions[
                self.chunks[1].pos_id]
            self.chunks[1].done =False

    def step(self):
        super().step()
        # Update chunk states
        self.init_pos_chunks()
        for chk in self.chunks:
            chk.nb_agents_touched = 0
            for ag in self.agents:
                if get_dist(ag.state.p_pos, chk.state.p_pos) <= \
                        chk.size + AGENT_RADIUS:
                    chk.nb_agents_touched += 1


class Scenario(BaseScenario):

    def make_world(self, nb_agents=4, obs_range=2.83, collision_pen=3.0,
                   chunk_radius=0.09):
        self.nb_agents = nb_agents
        self.obs_range = obs_range
        self.collision_pen = collision_pen
        self.agent_radius = AGENT_RADIUS
        self.chunk_radius = chunk_radius
        # Create world
        world = ForagingWorld(nb_agents, self.get_params())
        # Init world entities
        self.nb_agents = nb_agents
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.silent = True
            agent.size = AGENT_RADIUS
            agent.initial_mass = AGENT_MASS
            agent.accel = 3.5
            agent.color = np.array([0.0, 0.0, 0.0])
            agent.color += i / nb_agents
        # for chunk in world.chunks:
        #     chunk.size = self.chunk_radius
        world.chunks[0].size = self.chunk_radius - 0.02
        world.chunks[1].size = self.chunk_radius
        world.chunks[0].color = np.array([1.0, 1.0, 1.0])
        world.chunks[1].color = np.array([1.0, 0.0, 0.0])
        # Scenario attributes
        self.obs_range = obs_range
        # make initial conditions
        self.reset_world(world)
        return world
    
    def get_params(self):
        return {
            "nb_agents": self.nb_agents,
            "obs_range": self.obs_range,
            "collision_pen": self.collision_pen,
            "agent_radius": self.agent_radius,
            "chunk_radius": self.chunk_radius
        }

    def done(self, agent, world):
        return False

    def reset_world(self, world, seed=None, init_pos=None):
        if seed is not None:
            np.random.seed(seed)

        # Agents' initial pos
        # space = 2 / self.nb_agents
        # for i, agent in enumerate(world.agents):
        #     start = -1 + space * i
        #     end = start + space
        #     agent.state.p_pos = np.array([
        #         random.uniform(start + AGENT_RADIUS, end - AGENT_RADIUS),
        #         -1 + AGENT_RADIUS * 2])
        #     agent.state.c = np.zeros(world.dim_c)
        agent_positions = [
            np.array([-0.05, -0.05]),
            np.array([-0.05, 0.05]),
            np.array([0.05, -0.05]),
            np.array([0.05, 0.05])]
        for a_i, ag in enumerate(world.agents):
            ag.state.p_pos = agent_positions[a_i]
        # Buttons
        world.init_pos_chunks(True)
        for chk in world.chunks:
            chk.done = False
        # Set initial velocity
        for entity in world.entities:
            entity.state.p_vel = np.zeros(world.dim_p)

    def reward(self, agent, world):
        rew = 0.0
        if world.chunks[0].nb_agents_touched >= 1:
            rew += 2.0
            world.chunks[0].done = True
        if world.chunks[1].nb_agents_touched >= 2:
            rew += 100.0
            world.chunks[1].done = True
        return rew

    def observation(self, agent, world):
        obs = [agent.state.p_pos, agent.state.p_vel]

        for ag in world.agents:
            if ag is agent: continue
            if get_dist(agent.state.p_pos, ag.state.p_pos) <= self.obs_range:
                obs.append(np.concatenate((
                    [1.0],
                    (ag.state.p_pos - agent.state.p_pos) / self.obs_range, # Relative position normailised into [0, 1]
                    ag.state.p_vel # Velocity
                )))
            else:
                obs.append(np.array([0.0, 1.0, 1.0, 0.0, 0.0]))
        for c in world.chunks:
            if get_dist(agent.state.p_pos, c.state.p_pos) <= self.obs_range:
                obs.append(np.concatenate((
                    [1.0], 
                    (c.state.p_pos - agent.state.p_pos) / self.obs_range, # Relative position normailised into [0, 1]
                )))
            else:
                obs.append(np.array([0.0, 1.0, 1.0]))

        return np.concatenate(obs)