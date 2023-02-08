import numpy as np
import random

from multiagent.scenario import BaseScenario
from multiagent.core import Walled_World, Agent, Landmark, Action, Entity

BUTTON_RADIUS = 0.05
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
        self.color_name = None

    def is_pushing(self, agent_pos):
        return get_dist(agent_pos, self.state.p_pos) < BUTTON_RADIUS


class PushButtons(Walled_World):

    colors = {
        "red": np.array([1.0, 0.0, 0.0]),
        "green": np.array([0.0, 1.0, 0.0]),
        "blue": np.array([0.0, 0.0, 1.0]),
        "yellow": np.array([1.0, 1.0, 0.0]),
        "pink": np.array([0.0, 1.0, 1.0])
    }

    def __init__(self, nb_agents=2, nb_buttons=3):
        super(PushButtons, self).__init__()
        # add agent
        self.nb_agents = nb_agents
        self.agents = [Agent() for i in range(nb_agents)]
        # Buttons
        self.nb_buttons = nb_buttons
        self.buttons = [Button() for i in range(nb_buttons * 2)]
        self.colors_pushed = {
            self.colors.keys()[i]: 0 for i in range(nb_buttons)}
        # Control inertia
        self.damping = 0.8
        # Global reward at each step
        self.global_reward = 0.0

    @property
    def entities(self):
        return self.agents + self.buttons

    def step(self):
        super().step()
        self.colors_pushed = {
            self.colors.keys()[i]: 0 for i in range(nb_buttons)}
        for b in self.buttons:
            for a in self.agents:
                if b.is_pushing(a.state.p_pos):
                    self.colors_pushed[b.color_name] += 1
                    break


class Scenario(BaseScenario):

    def make_world(self, nb_agents=2, nb_buttons=3, obs_range=2.83, 
                   collision_pen=3.0):
        world = PushButtons(nb_agents, nb_buttons)
        # Init world entities
        self.nb_agents = nb_agents
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.silent = True
            agent.size = AGENT_RADIUS
            agent.initial_mass = AGENT_MASS
            agent.accel = 4.0
            agent.color = np.array([0.0, 0.0, 0.0])
            agent.color += i / nb_agents
        for button in world.buttons:
            button.size = BUTTON_RADIUS
        # Scenario attributes
        self.obs_range = obs_range
        # Reward attributes
        self.collision_pen = collision_pen
        # make initial conditions
        self.reset_world(world)
        return world

    def done(self, agent, world):
        return False

    def reset_world(self, world, seed=None, init_pos=None):
        if seed is not None:
            np.random.seed(seed)

        # Agents' initial pos
        agent_positions = [
            [-0.2, 0.0],
            [0.0, 0.0],
            [0.2, 0.0]]
        for i, agent in enumerate(world.agents):
            agent.state.p_pos = np.array(agent_positions[i])
            agent.state.c = np.zeros(world.dim_c)
        # Buttons
        button_positions = [-0.5, 0.0, 0.5]
        for i, button in enumerate(world.buttons):
            y = -0.5 if i < 3 else 0.5
            button.state.p_pos = np.array([button_positions[i % 3], y])
            button.color_name = world.colors.keys()[random.randint(0, 2)]
            button.color = world.colors[button.color_name]
        # Button's initial pos
        if init_pos is None:
            world.button.state.p_pos = np.array([
                random.uniform(-1 + BUTTON_RADIUS, 1 - BUTTON_RADIUS),
                1 - BUTTON_RADIUS])
        else:
            world.button.state.p_pos = np.array(init_pos["button"])
        # Set initial velocity
        for entity in world.entities:
            entity.state.p_vel = np.zeros(world.dim_p)
        # Initialise state of button and wall
        world.button.pushed = False
        self._done_flag = False

    def reward(self, agent, world):
        rew = -self.step_penalty
        
        # Reward if task complete
        dists = [get_dist(obj.state.p_pos, world.landmarks[i].state.p_pos)
                 for i, obj in enumerate(world.objects)]
        if not self._done_flag:
            self._done_flag = all(d <= LANDMARK_RADIUS for d in dists)
            if self._done_flag:
                world.global_reward += self.reward_done

        rew += world.global_reward

        # Penalty for collision between agents
        if agent.collide:
            for other_agent in world.agents:
                if other_agent is agent: continue
                dist = get_dist(agent.state.p_pos, other_agent.state.p_pos)
                dist_min = agent.size + other_agent.size
                if dist <= dist_min:
                    rew -= self.collision_pen
        return rew

    def observation(self, agent, world):
        """
        Observation:
         - Agent state: position, velocity
         - Other agents: [distance x, distance y, v_x, v_y]
         - Objects:
            - If in sight: [1, distance x, distance y, v_x, v_y]
            - If not: [0, 0, 0, 0, 0]
         - Landmarks:
            - If in sight: [1, distance x, distance y]
            - If not: [0, 0, 0]
         - Button:
            - If in sight: [1, state, distance x, distance y]
            - If not: [0, 0, 0, 0]
        => Full observation dim = 
            2 + 2 + 4 x (nb_agents - 1) + 5 x nb_objects + 3 x nb_landmarks + 4
        All distances are divided by max_distance to be in [0, 1]
        """
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
        for obj in world.objects:
            if get_dist(agent.state.p_pos, obj.state.p_pos) <= self.obs_range:
                obs.append(np.concatenate((
                    [1.0], # Bit saying entity is observed
                    (obj.state.p_pos - agent.state.p_pos) / self.obs_range, # Relative position normalised into [0, 1]
                    obj.state.p_vel # Velocity
                )))
            else:
                obs.append(np.array([0.0, 1.0, 1.0, 0.0, 0.0]))
        for lm in world.landmarks:
            if get_dist(agent.state.p_pos, lm.state.p_pos) <= self.obs_range:
                obs.append(np.concatenate((
                    [1.0], 
                    (lm.state.p_pos - agent.state.p_pos) / self.obs_range, # Relative position normailised into [0, 1]
                )))
            else:
                obs.append(np.array([0.0, 1.0, 1.0]))
        if get_dist(
            agent.state.p_pos, world.button.state.p_pos) <= self.obs_range:
            obs.append(np.concatenate((
                [1.0, float(world.button.pushed)], 
                (world.button.state.p_pos - agent.state.p_pos) / self.obs_range, # Relative position normailised into [0, 1]
            )))
        else:
            obs.append(np.array([0.0, 0.0, 1.0, 1.0]))

        return np.concatenate(obs)