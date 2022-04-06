import numpy as np

from offpolicy.envs.mpe.scenario import BaseScenario
from offpolicy.envs.mpe.core import World, Agent, Landmark, Action, Entity

LANDMARK_SIZE = 0.03
OBJECT_SIZE = 0.16
OBJECT_MASS = 2.0
AGENT_SIZE = 0.05
AGENT_MASS = 0.4

def get_dist(pos1, pos2, squared=False):
    dist = np.sum(np.square(pos1 - pos2))
    if squared:
        return dist
    else:
        return np.sqrt(dist)

def obj_callback(agent, world):
    action = Action()
    action.u = np.zeros((world.dim_p))
    action.c = np.zeros((world.dim_c))
    return action

class Object(Entity):
    def __init__(self):
        super(Object, self).__init__()
        # Objects are movable
        self.movable = True

class PushWorld(World):
    def __init__(self, nb_objects):
        super(PushWorld, self).__init__()
        # List of objects to push
        self.nb_objects = nb_objects
        self.objects = [Object() for _ in range(self.nb_objects)]
        self.landmarks = [Landmark() for _ in range(self.nb_objects)]

    @property
    def entities(self):
        return self.agents + self.objects + self.landmarks

    def reset(self):
        for i in range(self.nb_objects):
            self.init_object(i)

    def init_object(self, obj_i, min_dist=0.9, max_dist=1.5):
        # Random color for both entities
        color = np.random.uniform(0, 1, self.dim_color)
        # Object
        self.objects[obj_i].name = 'object %d' % len(self.objects)
        self.objects[obj_i].color = color
        self.objects[obj_i].size = OBJECT_SIZE
        self.objects[obj_i].initial_mass = OBJECT_MASS
        # Landmark
        self.landmarks[obj_i].name = 'landmark %d' % len(self.landmarks)
        self.landmarks[obj_i].collide = False
        self.landmarks[obj_i].color = color
        self.landmarks[obj_i].size = LANDMARK_SIZE
        # Set initial positions
        if min_dist is not None:
            while True:
                self.objects[obj_i].state.p_pos = np.random.uniform(-1, 1, self.dim_p)
                self.landmarks[obj_i].state.p_pos = np.random.uniform(-1, 1, self.dim_p)
                dist = get_dist(self.objects[obj_i].state.p_pos, 
                                self.landmarks[obj_i].state.p_pos)
                if dist > min_dist and dist < max_dist:
                    break

    def step(self):
        super().step()
        # Randomly add an object
        #if np.random.random() < self.obj_prob:
        #    self.add_object_and_landmark()
        

class Scenario(BaseScenario):

    def make_world(self, nb_agents=4, nb_objects=1, obs_range=0.4, 
                   collision_pen=10.0, relative_coord=True, dist_reward=False):
        world = PushWorld(nb_objects)
        # add agent
        self.nb_agents = nb_agents
        world.agents = [Agent() for i in range(self.nb_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.silent = True
            agent.size = AGENT_SIZE
            agent.initial_mass = AGENT_MASS
            agent.color = np.array([0.5,0.0,0.0])
        # Objects and landmarks
        self.nb_objects = nb_objects
        # Scenario attributes
        self.obs_range = obs_range
        self.relative_coord = relative_coord
        self.dist_reward = dist_reward
        # Reward attributes
        self.collision_pen = collision_pen
        # Flag for end of episode
        self._done_flag = False
        # make initial conditions
        self.reset_world(world)
        return world

    def done(self, agent, world):
        # Done if all objects are on their landmarks
        return self._done_flag

    def reset_world(self, world):
        world.reset()
        # set initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, 1, world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        # Set initial velocity
        for entity in world.entities:
            entity.state.p_vel = np.zeros(world.dim_p)
        self._done_flag = False

    def reward(self, agent, world):
        # Reward = -1 x squared distance between objects and corresponding landmarks
        dists = [get_dist(obj.state.p_pos, 
                          world.landmarks[i].state.p_pos,
                          squared=True)
                    for i, obj in enumerate(world.objects)]
        rew = -sum(dists)
        # Check if done
        self._done_flag = all(d <= LANDMARK_SIZE ** 2 for d in dists)


        # Reward based on distance to object
        if self.dist_reward:
            mean_dist = sum([get_dist(agent.state.p_pos, world.objects[0].state.p_pos)
                                for agent in world.agents]) / self.nb_agents
            rew -= mean_dist

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
        # Observation:
        #  - Agent state: position, velocity
        #  - Other agents and objects:
        #     - If in sight: [1, relative x, relative y, v_x, v_y]
        #     - If not: [0, 0, 0, 0, 0]
        #  - Landmarks:
        #     - If in sight: [1, relative x, relative y]
        #     - If not: [0, 0, 0]
        # => Full observation dim = 2 + 2 + 5 x (nb_agents_objects - 1) + 3 x (nb_landmarks)
        # All distances are divided by max_distance to be in [0, 1]
        entity_obs = []
        for entity in world.agents + world.objects:
            if entity is agent: continue
            if get_dist(agent.state.p_pos, entity.state.p_pos) <= self.obs_range:
                # Pos: relative normalised
                #entity_obs.append(np.concatenate((
                #    [1.0], (entity.state.p_pos - agent.state.p_pos) / self.obs_range, entity.state.p_vel
                #)))
                # Pos: relative
                if self.relative_coord:
                    entity_obs.append(np.concatenate((
                        #[1.0], (entity.state.p_pos - agent.state.p_pos), entity.state.p_vel
                        (entity.state.p_pos - agent.state.p_pos), entity.state.p_vel
                    )))
                # Pos: absolute
                else:
                    entity_obs.append(np.concatenate((
                        #[1.0], entity.state.p_pos, entity.state.p_vel
                        entity.state.p_pos, entity.state.p_vel
                    )))
            else:
                entity_obs.append(np.zeros(4))
        for entity in world.landmarks:
            if get_dist(agent.state.p_pos, entity.state.p_pos) <= self.obs_range:
                # Pos: relative normalised
                #entity_obs.append(np.concatenate((
                #    [1.0], (entity.state.p_pos - agent.state.p_pos) / self.obs_range
                #)))
                # Pos: relative
                if self.relative_coord:
                    entity_obs.append(np.concatenate((
                        #[1.0], (entity.state.p_pos - agent.state.p_pos)
                        (entity.state.p_pos - agent.state.p_pos)
                    )))
                # Pos: absolute
                else:
                    # entity_obs.append(np.concatenate((
                    #     [1.0], entity.state.p_pos
                    # )))
                    entity_obs.append(entity.state.p_pos)
            else:
                entity_obs.append(np.zeros(2))

        # Communication


        return np.concatenate([agent.state.p_pos, agent.state.p_vel] + entity_obs)