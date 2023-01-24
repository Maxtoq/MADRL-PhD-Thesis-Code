import numpy as np
import random

from multiagent.scenario import BaseScenario
from multiagent.core import Walled_World, Agent, Landmark, Action, Entity

BUTTON_RADIUS = 0.06
LANDMARK_RADIUS = 0.8 #1
OBJECT_RADIUS = 0.15 #0.3
OBJECT_MASS = 0.8
AGENT_RADIUS = 0.045
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
        self.pushed = False

    def is_pushing(self, agent_pos):
        return get_dist(agent_pos, self.state.p_pos) < BUTTON_RADIUS

class Object(Entity):
    def __init__(self):
        super(Object, self).__init__()
        # Objects are movable
        self.movable = True

class ClickNPushWorld(Walled_World):
    def __init__(self, nb_agents=3):
        super(ClickNPushWorld, self).__init__()
        # add agent
        self.nb_agents = nb_agents
        self.agents = [Agent() for i in range(self.nb_agents)]
        # Object
        self.object = Object()
        # Landmark
        self.landmark = Landmark()
        # Buttons
        self.buttons = [Button() for i in range(2)]
        # Control inertia
        self.damping = 0.8

    @property
    def entities(self):
        return self.agents + [self.object, self.landmark] + self.buttons

    def step(self):
        # last_obj_lm_dists = np.copy(self.obj_lm_dists)
        super().step()
        # Check if button is pushed to set movable state of objects
        buttons_pushed = [False, False]
        for i, b in enumerate(self.buttons):
            b.pushed = False
            for a in self.agents:
                if b.is_pushing(a.state.p_pos):
                    buttons_pushed[i] = True
                    b.pushed = True
                    break
        object_move = all(buttons_pushed)
        if object_move:
            self.object.movable = True


class Scenario(BaseScenario):

    def make_world(self, nb_agents=3, nb_objects=1, obs_range=2.83, 
                   collision_pen=15.0, reward_done=500, step_penalty=5.0, 
                   reward_buttons_pushed=4.9):
        world = ClickNPushWorld(nb_agents)
        # Agents
        self.nb_agents = nb_agents
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.silent = True
            agent.size = AGENT_RADIUS
            agent.initial_mass = AGENT_MASS
            agent.accel = 4.0
            agent.color = np.array([0.0, 0.0, 0.0])
            agent.color[i % 3] = 1.0
        # Object
        obj_color = np.random.uniform(0, 1, world.dim_color)
        world.object.size = OBJECT_RADIUS
        world.object.initial_mass = OBJECT_MASS
        world.object.color = obj_color
        # Landmark
        world.landmark.size = LANDMARK_RADIUS
        world.landmark.collide = False
        world.landmark.color = obj_color
        # Buttons
        button_color = np.random.uniform(0, 1, world.dim_color)
        for b in world.buttons:
            b.size = BUTTON_RADIUS
            b.color = button_color
        # Scenario attributes
        self.obs_range = obs_range
        # Reward attributes
        self.collision_pen = collision_pen
        # Flag for end of episode
        self._done_flag = False
        # Reward for completing the task
        self.reward_done = reward_done
        # Penalty for step in the environment
        self.step_penalty = step_penalty
        # Reward for pushing all buttons
        self.reward_buttons_pushed = reward_buttons_pushed
        # make initial conditions
        self.reset_world(world)
        return world

    def done(self, agent, world):
        return self._done_flag

    def reset_world(self, world, seed=None, init_pos=None):
        if seed is not None:
            np.random.seed(seed)

        # Check if init positions are valid
        if init_pos is not None:
            if (len(init_pos["agents"]) != self.nb_agents or 
                len(init_pos["objects"]) != self.nb_objects):
                print("ERROR: The initial positions {} are not valid.".format(
                    init_pos))
                exit(1)

        # Agents' initial pos
        world.agents[0].state.p_pos = np.array([-0.75, 1 - AGENT_RADIUS])
        world.agents[1].state.p_pos = np.array([0.0, 1 - AGENT_RADIUS])
        world.agents[2].state.p_pos = np.array([0.75, 1 - AGENT_RADIUS])
        # for i, agent in enumerate(world.agents):
        #     if init_pos is None:
        #         agent.state.p_pos = np.array([
        #             random.uniform(-1 + agent.size, 1 - agent.size),
        #             random.uniform(-1 + agent.size, 1 - agent.size)
        #         ])
        #     else:
        #         agent.state.p_pos = np.array(init_pos["agents"][i])
        #     agent.state.c = np.zeros(world.dim_c)
        # Object and landmark
        world.object.movable = False
        world.object.state.p_pos = np.array([0.0, 0.5])
        world.landmark.state.p_pos = np.array([0.0, -1.0])
        # Buttons
        world.buttons[0].state.p_pos = np.array([-0.5, 0.0])
        world.buttons[1].state.p_pos = np.array([0.5, 0.0])
        # Set initial velocity
        for entity in world.entities:
            entity.state.p_vel = np.zeros(world.dim_p)
        # Initialise state of button and wall
        for b in world.buttons:
            b.pushed = False
        self._done_flag = False

    def reward(self, agent, world):
        rew = -self.step_penalty
        
        # Reward if task complete
        dist = get_dist(world.object.state.p_pos, world.landmark.state.p_pos)
        if not self._done_flag:
            self._done_flag = dist <= LANDMARK_RADIUS
        if self._done_flag:
            rew += self.reward_done

        # Reward if all buttons pushed
        if all([b.pushed for b in world.buttons]):
            rew += self.reward_buttons_pushed

        # Penalty for collision between agents
        # if agent.collide:
        #     for other_agent in world.agents:
        #         if other_agent is agent: continue
        #         dist = get_dist(agent.state.p_pos, other_agent.state.p_pos)
        #         if dist <= agent.size + other_agent.size:
        #             rew -= self.collision_pen
        return rew

    def observation(self, agent, world):
        """
        Observation:
         - Agent state: position, velocity
         - Other agents: [distance x, distance y, v_x, v_y]
         - Object:
            - If in sight: [1, distance x, distance y, v_x, v_y]
            - If not: [0, 0, 0, 0, 0]
         - Landmark:
            - If in sight: [1, distance x, distance y]
            - If not: [0, 0, 0]
         - Buttons:
            - If in sight: [1, state, distance x, distance y]
            - If not: [0, 0, 0, 0]
        => Full observation dim = 
            2 + 2 + 4 x (nb_agents - 1) + 5 x nb_object + 3 x nb_landmark + 4 x nb_button
        All distances are divided by max_distance to be in [0, 1]
        """
        obs = [agent.state.p_pos, agent.state.p_vel]
        # Other agents
        for ag in world.agents:
            if ag is agent: continue
            if get_dist(agent.state.p_pos, ag.state.p_pos) <= self.obs_range:
                obs.append(np.concatenate((
                    [1.0], # Bit saying entity is observed
                    (ag.state.p_pos - agent.state.p_pos) / self.obs_range, # Relative position normailised into [0, 1]
                    ag.state.p_vel # Velocity
                )))
            else:
                obs.append(np.array([0.0, 1.0, 1.0, 0.0, 0.0]))
        # Object
        if get_dist(agent.state.p_pos, world.object.state.p_pos) <= self.obs_range:
            obs.append(np.concatenate((
                [1.0], # Bit saying entity is observed
                (world.object.state.p_pos - agent.state.p_pos) / self.obs_range, # Relative position normalised into [0, 1]
                world.object.state.p_vel # Velocity
            )))
        else:
            obs.append(np.array([0.0, 1.0, 1.0, 0.0, 0.0]))
        # Landmark
        if get_dist(agent.state.p_pos, world.landmark.state.p_pos) <= self.obs_range:
            obs.append(np.concatenate((
                [1.0], 
                (world.landmark.state.p_pos - agent.state.p_pos) / self.obs_range, # Relative position normailised into [0, 1]
            )))
        else:
            obs.append(np.array([0.0, 1.0, 1.0]))
        # Buttons
        for b in world.buttons:
            if get_dist(
                agent.state.p_pos, b.state.p_pos) <= self.obs_range:
                obs.append(np.concatenate((
                    [1.0, float(b.pushed)], 
                    (b.state.p_pos - agent.state.p_pos) / self.obs_range, # Relative position normailised into [0, 1]
                )))
            else:
                obs.append(np.array([0.0, 0.0, 1.0, 1.0]))

        return np.concatenate(obs)