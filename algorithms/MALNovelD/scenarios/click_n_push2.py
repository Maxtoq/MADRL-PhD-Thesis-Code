import numpy as np
import random

from multiagent.scenario import BaseScenario
from multiagent.core import World, Agent, Landmark, Action, Entity

BUTTON_RADIUS = 0.03
LANDMARK_RADIUS = 0.05
OBJECT_RADIUS = 0.15
OBJECT_MASS = 2.0
AGENT_RADIUS = 0.04
AGENT_MASS = 0.4

def get_dist(pos1, pos2, squared=False):
    dist = np.sum(np.square(pos1 - pos2))
    if squared:
        return dist
    else:
        return np.sqrt(dist)


class Wall:

    def __init__(self, orientation, position):
        if orientation == "hor":
            self.orient = 1
        elif orientation == "ver":
            self.orient = 0
        else:
            print("ERROR: orientation parameter must be 'hor' or 'ver'.")
            exit()

        if not -1 <= position <= 1:
            print("ERROR: position parameter must be in [-1, 1]- interval.")
            exit()
        self.position = position
        # State
        self.active = True
        
    def block(self, entity, temp_pos):
        if not self.active:
            return
        if entity.state.p_pos[self.orient] > self.position:
            if temp_pos[self.orient] - entity.size < self.position:
                entity.state.p_vel[self.orient] = 0.0
                entity.state.p_pos[self.orient] = self.position + entity.size
        elif entity.state.p_pos[self.orient] < self.position:
            if temp_pos[self.orient] + entity.size > self.position:
                entity.state.p_vel[self.orient] = 0.0
                entity.state.p_pos[self.orient] = self.position - entity.size

class Button(Landmark):

    def __init__(self):
        super(Button, self).__init__()
        self.collide = False

    def is_pushing(self, agent_pos):
        return get_dist(agent_pos, self.state.p_pos) < BUTTON_RADIUS

class Object(Entity):
    def __init__(self):
        super(Object, self).__init__()
        # Objects are movable
        self.movable = True

class ClickNPushWorld(World):
    def __init__(self, nb_agents, nb_objects):
        super(ClickNPushWorld, self).__init__()
        # add agent
        self.nb_agents = nb_agents
        self.agents = [Agent() for i in range(self.nb_agents)]
        # Object
        self.nb_objects = nb_objects
        self.objects = [Object() for i in range(self.nb_objects)]
        # Corresponding landmarks
        self.landmarks = [Landmark() for _ in range(self.nb_objects)]
        # Distances between objects and their landmark
        self.obj_lm_dists = np.zeros(self.nb_objects)
        # Button
        self.button = Button()
        # Control inertia
        self.damping = 0.8
        # Add walls on each side and 
        self.walls = {
            "South": Wall("ver", -1),
            "North": Wall("ver", 1),
            "West": Wall("hor", -1),
            "East": Wall("hor", 1)
        }
        # Global reward at each step
        self.global_reward = 0.0

    @property
    def entities(self):
        return self.agents + self.objects + self.landmarks + [self.button]

    def step(self):
        last_obj_lm_dists = np.copy(self.obj_lm_dists)
        super().step()
        self.global_reward = 0.0
        # Compute shaping reward
        for obj_i in range(self.nb_objects):
            # Update dists
            self.obj_lm_dists[obj_i] = get_dist(
                self.objects[obj_i].state.p_pos,
                self.landmarks[obj_i].state.p_pos)
            # Compute reward
            self.global_reward += 100 * (
                last_obj_lm_dists[obj_i] - self.obj_lm_dists[obj_i])
        # Check if button is pushed to set movable state of objects
        objects_move = False
        for ag in self.agents:
            if self.button.is_pushing(ag.state.p_pos):
                objects_move = True
                self.global_reward += 0.1
                break
        for obj in self.objects:
            obj.movable = objects_move

    # Integrate state with walls blocking entities on each side
    def integrate_state(self, p_force):
        for i,entity in enumerate(self.entities):
            if not entity.movable: continue
            entity.state.p_vel = entity.state.p_vel * (1 - self.damping)
            if (p_force[i] is not None):
                entity.state.p_vel += (p_force[i] / entity.mass) * self.dt
            if entity.max_speed is not None:
                speed = np.sqrt(np.square(entity.state.p_vel[0]) \
                        + np.square(entity.state.p_vel[1]))
                if speed > entity.max_speed:
                    entity.state.p_vel = entity.state.p_vel / \
                        np.sqrt(np.square(entity.state.p_vel[0]) +
                            np.square(entity.state.p_vel[1])) * entity.max_speed
            # Check for wall collision
            temp_pos = entity.state.p_pos + entity.state.p_vel * self.dt
            for wall in self.walls.values():
                wall.block(entity, temp_pos)
            entity.state.p_pos += entity.state.p_vel * self.dt

class Scenario(BaseScenario):

    def make_world(self, nb_agents=2, nb_objects=1, obs_range=2.83, 
                   collision_pen=1, reward_done=50, reward_button_pushed=10, 
                   step_penalty=0.1, obj_lm_dist_range=[0.3, 1.5]):
        world = ClickNPushWorld(nb_agents, nb_objects)
        # Init world entities
        self.nb_agents = nb_agents
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.silent = True
            agent.size = AGENT_RADIUS
            agent.initial_mass = AGENT_MASS
            agent.color = np.array([0.0, 0.0, 0.0])
            agent.color[i % 3] = 1.0
        self.nb_objects = nb_objects
        for i, obj in enumerate(world.objects):
            color = np.random.uniform(0, 1, world.dim_color)
            obj.size = OBJECT_RADIUS
            obj.initial_mass = OBJECT_MASS
            obj.color = color
            # Corresponding Landmarks
            world.landmarks[i].size = LANDMARK_RADIUS
            world.landmarks[i].collide = False
            world.landmarks[i].color = color
        world.button.size = BUTTON_RADIUS
        world.button.color = np.random.uniform(0, 1, world.dim_color)
        # Scenario attributes
        self.obs_range = obs_range
        self.obj_lm_dist_range = obj_lm_dist_range
        # Reward attributes
        self.collision_pen = collision_pen
        # Flag for end of episode
        self._done_flag = False
        # Reward for completing the task
        self.reward_done = reward_done
        self.reward_button_pushed = reward_button_pushed
        # Penalty for step in the environment
        self.step_penalty = step_penalty
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
        for i, agent in enumerate(world.agents):
            if init_pos is None:
                agent.state.p_pos = np.array([
                    random.uniform(-1 + agent.size, 1 - agent.size),
                    random.uniform(-1 + agent.size, 1 - 2 * agent.size)
                ])
            else:
                agent.state.p_pos = np.array(init_pos["agents"][i])
            agent.state.c = np.zeros(world.dim_c)
        # Objects and landmarks
        for i, obj in enumerate(world.objects):
            # Initial unmovable state
            obj.movable = False
            # Positions
            if init_pos is None:
                while True:
                    obj.state.p_pos = np.array([
                        random.uniform(-1 + obj.size, 1 - obj.size),
                        random.uniform(-1 + obj.size, 1 - 2 * obj.size)])
                    world.landmarks[i].state.p_pos = np.array([
                        random.uniform(-1 + obj.size, 1 - obj.size),
                        random.uniform(-1 + obj.size, 1 - 2 * obj.size)])
                    dist = get_dist(
                        obj.state.p_pos, world.landmarks[i].state.p_pos)
                    if (self.obj_lm_dist_range is None  or 
                        (dist > self.obj_lm_dist_range[0] and 
                         dist < self.obj_lm_dist_range[1])):
                        break
            else:
                obj.state.p_pos = np.array(init_pos["objects"][i])
                world.landmarks[i].state.p_pos = np.array(
                    init_pos["landmarks"][i])
                dist = get_dist(
                    obj.state.p_pos, world.landmarks[i].state.p_pos)
            # Set distances between objects and their landmark
            world.obj_lm_dists[i] = dist
        # Button's initial pos
        if init_pos is None:
            world.button.state.p_pos = np.array([
                random.uniform(-1 + BUTTON_RADIUS, 1 - BUTTON_RADIUS),
                1 - AGENT_RADIUS])
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
        => Full observation dim = 2 + 2 + 4 x (nb_agents - 1) + 5 x (nb_objects) + 4
        All distances are divided by max_distance to be in [0, 1]
        """
        obs = [agent.state.p_pos, agent.state.p_vel]

        for ag in world.agents:
            if ag is agent: continue
            obs.append(np.concatenate((
                (ag.state.p_pos - agent.state.p_pos) / 2.83, # Relative position normailised into [0, 1]
                ag.state.p_vel # Velocity
            )))
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