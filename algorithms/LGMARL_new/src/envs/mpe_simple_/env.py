import numpy as np
import random

from ..mpe.core import Agent, Action, Walled_World
from ..mpe.scenario import BaseScenario


AGENT_RADIUS = 0.12
AGENT_MASS = 0.5

OBS_RANGE = 0.6

REWARD_CAPTURE = 30.0
PENALTY_MISS = 0.0
PENALTY_STEP = 1.0


def get_dist(pos1, pos2, squared=False):
    dist = np.sum(np.square(pos1 - pos2))
    if squared:
        return dist
    else:
        return np.sqrt(dist)


class Prey(Agent):

    def __init__(self):
        super(Prey, self).__init__()
        self.silent = True
        self.blind = True
        # random actions
        self.action_callback = self._rand_action
        # Status
        self.caught = False
        self.visible = True

    def _rand_action(self, world):
        a = Action()
        a.u = np.random.uniform(-4, 4, (2))
        return a
    
    def catch(self):
        self.caught = True
        self.visible = False
        self.collide = False
        self.movable = False

    def reset(self):
        self.caught = False
        self.visible = True
        self.collide = True
        self.movable = True
    

class PredPreyWorld(Walled_World):

    def __init__(self, n_agents, n_preys):
        super(PredPreyWorld, self).__init__()
        self.n_agents = n_agents
        self.n_preys = n_preys

        self.agents = []
        self.preys = []
        for a_i in range(self.n_agents):
            agent = Agent()
            agent.name = 'agent %d' % a_i
            agent.silent = True
            agent.size = AGENT_RADIUS
            agent.initial_mass = AGENT_MASS
            agent.color = np.array([0.0,0.0,1.0])
            self.agents.append(agent)
        # Preys
        for p_i in range(self.n_preys):
            prey = Prey()
            prey.name = 'prey %d' % p_i
            prey.size = AGENT_RADIUS
            prey.initial_mass = AGENT_MASS
            prey.color = np.array([1.0,0.0,0.0])
            self.preys.append(prey)

        # Flag for rewarding a catch
        self.catch_reward = 0

        # Full obs
        self.full_obs = np.zeros((20, 20, 3))

        self.current_step = 0

    @property
    def entities(self):
        return self.agents + self.preys + self.landmarks

    @property
    def policy_agents(self):
        return self.agents

    @property
    def scripted_agents(self):
        return self.preys
    
    def step(self):
        super(Walled_World, self).step()

        self.catch_reward = 0
        # Check for caught preys
        for p in self.preys:
            if p.caught:
                continue
            n_agent = 0
            for a in self.agents:
                if get_dist(p.state.p_pos, a.state.p_pos) <= p.size + a.size:
                    n_agent += 1
            if n_agent >= 2:
                p.catch()
                self.catch_reward +=1

        self.current_step += 1

        # Update full obs
        # self.full_obs = np.zeros((20, 20, 3))
        # for e in self.entities:
        #     pos = (e.state.p_pos + 1) * 10
        #     self.full_obs[int(pos[0]), int(pos[1])] = e.color


class Scenario(BaseScenario):

    def make_world(self, n_agents=4, n_preys=2, max_steps=100):
        self.n_agents, self.n_preys = n_agents, n_preys
        # self.obs_range = obs_range
        self.max_steps = max_steps

        self.world = PredPreyWorld(n_agents, n_preys)
        self.world.dim_c = 0 # No communication via mpe

        # make initial conditions
        self.reset_world()

    def seed(self, seed):
        np.random.seed(seed)
        random.seed(seed)

    def done(self, agent):
        # Done if all preys are caught
        return all([p.caught for p in self.world.preys]) \
                or self.world.current_step >= self.max_steps

    def reset_world(self, seed=None, init_pos=None):
        if seed is not None:
            np.random.seed(seed)

        # Check if init positions are valid
        if init_pos is not None:
            assert (len(init_pos["agents"]) != self.n_agents or 
                len(init_pos["preys"]) != self.n_preys), f"ERROR: The initial positions {init_pos} are not valid."

        # Agents' initial pos
        for a_i, agent in enumerate(self.world.agents):
            if init_pos is None:
                agent.state.p_pos = np.random.uniform(
                    -1 + agent.size, 1 - agent.size, self.world.dim_p)
            else:
                agent.state.p_pos = np.array(init_pos["agents"][a_i])

        # Preys' initial pos
        for p_i, prey in enumerate(self.world.preys):
            if init_pos is None:
                prey.state.p_pos = np.random.uniform(
                    -1 + prey.size, 1 - prey.size, self.world.dim_p)
            else:
                prey.state.p_pos = np.array(init_pos["preys"][p_i])
            prey.reset()

        # Set initial velocity
        for entity in self.world.entities:
            entity.state.p_vel = np.zeros(self.world.dim_p)

        self.world.current_step = 0

    def reward(self, agent):
        reward = self.world.catch_reward * REWARD_CAPTURE - PENALTY_STEP

        if PENALTY_MISS > 0:
            pass # TODO

        return reward
        
    def observation(self, agent):
        """
        Observation:
         - Agent state: position
         - Other agents and preys:
            - If in sight: [1, distance x, distance y, color]
            - If not: [0, 1, 1, 0, 0, 0]
        => Full observation dim = 2 + 6 x (n_agents - 1 + n_preys)
        All distances are divided by max_distance to be in [0, 1]
        """
        obs = [agent.state.p_pos]

        for e in self.world.entities:
            if e is agent: continue
            if get_dist(agent.state.p_pos, e.state.p_pos) <= OBS_RANGE:
                obs.append(np.concatenate((
                    [1.0],
                    (e.state.p_pos - agent.state.p_pos) / OBS_RANGE, # Relative position normailised into [0, 1]
                    e.color # Velocity
                )))
            else:
                obs.append(np.array([0.0, 1.0, 1.0, 0.0, 0.0, 0.0]))

        return np.concatenate(obs)

    


