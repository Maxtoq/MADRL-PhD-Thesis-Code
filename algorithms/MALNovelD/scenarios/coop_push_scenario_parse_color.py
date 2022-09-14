import numpy as np

from multiagent.scenario import BaseScenario
from multiagent.core import World, Agent, Action, Entity


# For parser
from utils.parsers import ColorParser
import random

LANDMARK_SIZE = 0.1
OBJECT_SIZE = 0.15
OBJECT_MASS = 1.0
AGENT_SIZE = 0.04
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

# --------- Parser ---------
# Color parsers
class ObservationParser(ColorParser):
    
    vocab = ['Located', 'Object', 'Landmark', 'North', 'South', 'East', 'West', 'Center', 'Not', "Red", "Blue", "Green"]
    
    def __init__(self, nb_agents, nb_objects, chance_not_sent):
        """
        ObservationParser, generate descriptions of the agents' observations.
        Inputs:
            nb_agents (int): Number of agents.
            nb_objects (int): Number of objects.
            chance_not_sent (float): Chance of generating a not sentence.
        """
        super(ObservationParser, self).__init__()
        self.nb_agents = nb_agents
        self.nb_objects = nb_objects
        self.chance_not_sent = chance_not_sent

    # Generate a sentence for the object
    def object_sentence(self, obs):
        '''
        Will generate a sentence if the agent sees the object
        with the position of this object and will generate another sentence
        if this agent is pushing the object
        Input:  
            obs: list(float) observation link to this object
        Output: 
            sentence: list(str) The object sentence generated
        '''
        sentence = []

        # If visible                                      
        if  obs[0] == 1 :
            sentence.append(self.array_to_color(obs[5:8]))
            sentence.append("Object")
            # North / South
            if  obs[2] >= 0.25:
                sentence.append("North")
            elif  obs[2] < -0.25:
                sentence.append("South")
            
            # West / East
            if  obs[1] >= 0.25:
                sentence.append("East")
            elif  obs[1] < -0.25:
                sentence.append("West")            

        return sentence

     # Generate a sentence for the landmark
    def landmark_sentence(self, obs):
        '''
        Will generate a sentence if the agent sees the landmark
        with the position of this landmark

        Input:  
            obs: list(float) observation link to this object (1: visible or not, 2: position)

        Output: sentence: list(str) The landmark sentence generated
        '''
        sentence = []
        # Variable to check if we are close to the landmark
        # = True until we check its position
        close = True

        # If visible
        if  obs[0] == 1 :
            sentence.append(self.array_to_color(obs[3:6]))
            sentence.append("Landmark")
            
            # North / South
            if  obs[2] >= 0.2:
                sentence.append("North")
                close = False
            elif  obs[2] < -0.2:
                sentence.append("South")
                close = False

            # West / East
            if  obs[1] >= 0.2:
                sentence.append("East")
                close = False
            elif  obs[1] < -0.2:
                sentence.append("West")
                close = False
            
            #If we are close to landmark
            if close:
                # North / South
                if  obs[2] >= 0:
                    sentence.append("North")
                elif  obs[2] < 0:
                    sentence.append("South")
                    
                # West / East
                if  obs[1] >= 0:
                    sentence.append("East")
                elif  obs[1] < 0:
                    sentence.append("West")

        return sentence

    # Might generate a not sentence
    def not_sentence(self, position, not_vis_obj, not_vis_land):
        '''
        Might Create a "Not sentence" if the agent don't see 1 
        or more types of object

        Input:  
            position: (list(float)) The position of the agent
            not_vis_obj: (list(int)) List of objects number not visible
            not_vis_land: (list(int)) List of landmarks number not visible

        Output: list(str) A sentence based on what it doesn't see
        '''
        sentence = []

        #Generation of a NOT sentence ?
        """
        if = 1: Will generate not_sentence only for objects
        if = 2: Will generate not_sentence only for landmarks
        if = 3: Will generate not_sentence for both objects and landmarks
        """
        not_sentence = 0
        # We don't always generate not sentence
        if random.random() <= self.chance_not_sent:
            not_sentence = random.randint(1,3)

            if not_sentence in [1, 3] and len(not_vis_obj) != 0:
                # Object not sentence
                # Pick a random object
                obj = random.choice(not_vis_obj)
                sentence.extend(
                    [obj.color_name, "Object", "Not"])
                for word in position:
                    sentence.append(word)
            if not_sentence in [2, 3] and len(not_vis_land) != 0:
                # Landmark not sentence
                # Pick a random object
                land = random.choice(not_vis_land)
                sentence.extend(
                    [land.color_name, "Landmark", "Not"])
                for word in position:
                    sentence.append(word)

        return sentence

    # Generate the full sentence for the agent
    def parse_obs(self, obs):
        '''
        Generate a description of an observation.
        Input:  
            obs: (np.array(float)) observation of the agent with all the 
                entities

        Output: sentence: list(str) The sentence generated
        '''
        # Sentence generated
        sentence = []
        # Position of the agent
        position = []

        # Get the position of the agent
        sentence.extend(self.position_agent(obs[0:2]))
        for i in range(1,len(sentence)):
            position.append(sentence[i])
        
        # Will calculate the place in the array of each entity
        # There are 4 values for the main agent
        """
        2: position
        2: velocity
        """
        place = 4

        # Add the values of the other agents
        """
        1: visible or not
        2: position
        2: velocity
        """
        for _ in range(self.nb_agents - 1):
            place += 5

        # Objects sentence
        """
        1: visible or not
        2: position
        2: velocity
        3: color
        """
        objects_sentence = []
        not_visible_obj = []
        for object in range(self.nb_objects):
            object_sentence = []
            object_sentence.extend(self.object_sentence(obs[place:place+8]))
            place += 8
            # If we don't see the object
            if len(object_sentence) == 0:
                # Get the number of the object
                not_visible_obj.append(object)
            # Else we append the sentence
            else:
                objects_sentence.extend(object_sentence)
        #no_objects = len(objects_sentence) == 0  
        sentence.extend(objects_sentence)

        # Landmarks sentence
        """
        1: visible or not
        2: position
        3: color
        """
        landmarks_sentence = []
        not_visible_land = []
        for landmark in range(self.nb_objects):
            landmark_sentence = []
            landmark_sentence.extend(
                self.landmark_sentence(obs[place:place + 6]))
            place += 6
            # If we don't see the object
            if len(landmark_sentence) == 0:
                # Get the number of the object
                not_visible_land.append(landmark)
            # Else we append the sentence
            else:
                landmarks_sentence.extend(landmark_sentence)
        #no_landmarks = len(landmarks_sentence) == 0
        sentence.extend(landmarks_sentence)

        # Not sentence
        sentence.extend(self.not_sentence(
            position, not_visible_obj, not_visible_land))

        return sentence

    # # Reset the informations on the agents
    # def reset(self, obj_colors, land_colors):
    #     """
    #     Reset the colors of the exercise
    #     """
    #     # Reset the colors
    #     self.obj_colors = obj_colors
    #     self.land_colors = land_colors

# --------- Scenario -----------
# All entities have a color and a shape
class Color_Entity(Entity):

    color_names = ["Red", "Blue", "Green"]
    colors = [
        [1, 0.22745, 0.18431], # Red
        [0, 0.38, 1], # Blue
        [0.2, 0.78, 0.35] # Green
    ]

    def __init__(self):
        super(Color_Entity, self).__init__()
        self.color_name = None
        self.color = None
        self.num_color = 0

    # Get the color based on the number
    def num_to_color(self, color_num):
        return self.colors[color_num]

    def num_to_color_name(self, color_num):
        return self.color_names[color_num]

# properties of landmark entities
class Landmark(Color_Entity):
     def __init__(self):
        super(Landmark, self).__init__()

# properties of object entities
class Object(Color_Entity):
    def __init__(self):
        super(Object, self).__init__()
        # Objects are movable
        self.movable = True

class PushWorld(World):
    def __init__(self, nb_agents, nb_objects):
        super(PushWorld, self).__init__()
        # add agent
        self.nb_agents = nb_agents
        self.agents = [Agent() for i in range(self.nb_agents)]
        # List of objects to push
        self.nb_objects = nb_objects
        self.objects = [Object() for _ in range(self.nb_objects)]
        # Corresponding landmarks
        self.landmarks = [Landmark() for _ in range(self.nb_objects)]
        # Distances between objects and their landmark
        self.obj_lm_dists = np.zeros(self.nb_objects)
        # Shaping reward based on distances between objects and lms
        self.shaping_reward = 0.0
        # Control inertia
        self.damping = 0.8

    @property
    def entities(self):
        return self.agents + self.objects + self.landmarks

    def reset(self):
        for i in range(self.nb_objects):
            self.init_object(i)

    def init_object(self, obj_i, min_dist=0.2, max_dist=1.5):
        # Random color for both entities
        color = np.random.uniform(0, 1, self.dim_color)
        #color = self.pick_color()
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
        # # Fixed initial pos
        # self.objects[obj_i].state.p_pos = np.zeros(2)
        # self.landmarks[obj_i].state.p_pos = np.array([-0.5, -0.5])
        # return
        if min_dist is not None:
            while True:
                self.objects[obj_i].state.p_pos = np.random.uniform(
                    -1 + OBJECT_SIZE, 1 - OBJECT_SIZE, self.dim_p)
                self.landmarks[obj_i].state.p_pos = np.random.uniform(
                    -1 + OBJECT_SIZE, 1 - OBJECT_SIZE, self.dim_p)
                dist = get_dist(self.objects[obj_i].state.p_pos, 
                                self.landmarks[obj_i].state.p_pos)
                if dist > min_dist and dist < max_dist:
                    break
        else:
            dist = get_dist(self.objects[obj_i].state.p_pos, 
                            self.landmarks[obj_i].state.p_pos)
        # Set distances between objects and their landmark
        self.obj_lm_dists[obj_i] = dist

    def step(self):
        # s
        last_obj_lm_dists = np.copy(self.obj_lm_dists)
        super().step()
        # s'
        # Compute shaping reward
        self.shaping_reward = 0.0
        for obj_i in range(self.nb_objects):
            # Update dists
            self.obj_lm_dists[obj_i] = get_dist(
                self.objects[obj_i].state.p_pos,
                self.landmarks[obj_i].state.p_pos)
            # Compute reward
            self.shaping_reward += last_obj_lm_dists[obj_i] \
                                    - self.obj_lm_dists[obj_i]

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
                    entity.state.p_vel = entity.state.p_vel / np.sqrt(np.square(entity.state.p_vel[0]) +
                                                                  np.square(entity.state.p_vel[1])) * entity.max_speed
            # Check for wall collision
            temp_pos = entity.state.p_pos + entity.state.p_vel * self.dt
            # West wall
            if temp_pos[0] - entity.size < -1:
                entity.state.p_vel[0] = 0.0
                entity.state.p_pos[0] = -1.0 + entity.size
            # East wall
            if temp_pos[0] + entity.size > 1:
                entity.state.p_vel[0] = 0.0
                entity.state.p_pos[0] = 1.0 - entity.size
            # North wall
            if temp_pos[1] - entity.size < -1:
                entity.state.p_vel[1] = 0.0
                entity.state.p_pos[1] = -1.0 + entity.size
            # South wall
            if temp_pos[1] + entity.size > 1:
                entity.state.p_vel[1] = 0.0
                entity.state.p_pos[1] = 1.0 - entity.size
            entity.state.p_pos += entity.state.p_vel * self.dt
                
        
class Scenario(BaseScenario):

    def make_world(self, nb_agents=4, nb_objects=1, obs_range=0.4,
                   collision_pen=1, relative_coord=True, dist_reward=False, 
                   reward_done=50, step_penalty=0.1, obj_lm_dist_range=[0.2, 1.5]):
        world = PushWorld(nb_agents, nb_objects)
        # add agent
        self.nb_agents = nb_agents
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.silent = True
            agent.size = AGENT_SIZE
            agent.initial_mass = AGENT_MASS
            agent.color = np.array([0.0,0.0,0.0])
            agent.color[i % 3] = 1.0
        # Objects and landmarks
        self.nb_objects = nb_objects
        for i, object in enumerate(world.objects):
            object.name = 'object %d' % i
            object.size = OBJECT_SIZE
            object.initial_mass = OBJECT_MASS
            # Corresponding landmarks
            world.landmarks[i].name = 'landmark %d' % i
            world.landmarks[i].collide = False
            world.landmarks[i].size = LANDMARK_SIZE
        
        # Orders of objects and landmarks in observation
        self.obj_order_in_obs = list(range(nb_objects))
        self.lm_order_in_obs = list(range(nb_objects))
        # Range of initial distance between objects and landmarks
        self.obj_lm_dist_range = obj_lm_dist_range
        # Scenario attributes
        self.obs_range = obs_range
        self.relative_coord = relative_coord
        self.dist_reward = dist_reward
        # Reward attributes
        self.collision_pen = collision_pen
        # Flag for end of episode
        self._done_flag = False
        # Reward for completing the task
        self.reward_done = reward_done
        # Penalty for step in the environment
        self.step_penalty = step_penalty
        # make initial conditions
        self.reset_world(world)
        return world


    def done(self, agent, world):
        # Done if all objects are on their landmarks
        return self._done_flag

    def reset_world(self, world, seed=None, init_pos=None):
        # Set seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Check if init positions are valid
        if init_pos is not None:
            if (len(init_pos["agents"]) != self.nb_agents or 
                len(init_pos["objects"]) != self.nb_objects or
                len(init_pos["landmarks"]) != self.nb_objects):
                print("ERROR: The initial positions {} are not valid.".format(
                    init_pos))
                exit(1)

        # Agents' initial pos
        for i, agent in enumerate(world.agents):
            if init_pos is None:
                agent.state.p_pos = np.random.uniform(
                    -1 + agent.size, 1 - agent.size, world.dim_p)
            else:
                agent.state.p_pos = np.array(init_pos["agents"][i])
            agent.state.c = np.zeros(world.dim_c)
            # Initial velocity
            agent.state.p_vel = np.zeros(world.dim_p)

        # Objects and landmarks' colors and initial pos 
        color_nums = random.sample(
            list(range(self.nb_objects)), self.nb_objects)
        for i, object in enumerate(world.objects):
            # Colors
            color_num = color_nums.pop(0)
            object.num_color = color_num
            object.color = object.num_to_color(color_num)
            object.color_name = object.num_to_color_name(color_num)
            # Corresponding landmarks
            world.landmarks[i].num_color = color_num
            world.landmarks[i].color = object.color
            world.landmarks[i].color_name = object.color_name
            
            # Positions
            if init_pos is None:
                while True:
                    object.state.p_pos = np.random.uniform(
                        -1 + OBJECT_SIZE, 1 - OBJECT_SIZE, world.dim_p)
                    world.landmarks[i].state.p_pos = np.random.uniform(
                        -1 + OBJECT_SIZE, 1 - OBJECT_SIZE, world.dim_p)
                    dist = get_dist(
                        object.state.p_pos, 
                        world.landmarks[i].state.p_pos)
                    if (self.obj_lm_dist_range is None  or 
                        (dist > self.obj_lm_dist_range[0] and 
                        dist < self.obj_lm_dist_range[1])):
                        break
            else:
                object.state.p_pos = np.array(init_pos["objects"][i])
                world.landmarks[i].state.p_pos = np.array(
                    init_pos["landmarks"][i])
                dist = get_dist(object.state.p_pos, 
                                world.landmarks[i].state.p_pos)
            # Set distances between objects and their landmark
            world.obj_lm_dists[i] = dist
            # Initial velocity
            object.state.p_vel = np.zeros(world.dim_p)
            world.landmarks[i].state.p_vel = np.zeros(world.dim_p)
        
        # Set order of objects and landmarks in observation
        self.obj_order_in_obs = random.sample(
            list(range(self.nb_objects)), self.nb_objects)
        self.lm_order_in_obs = random.sample(
            list(range(self.nb_objects)), self.nb_objects)

        self._done_flag = False

    def reward(self, agent, world):
        # Reward = -1 x squared distance between objects and corresponding landmarks
        dists = [get_dist(obj.state.p_pos, 
                          world.landmarks[i].state.p_pos)
                    for i, obj in enumerate(world.objects)]

        # Shaped reward
        shaped = 100 * world.shaping_reward
        rew = -self.step_penalty + shaped

        # Reward if task complete
        self._done_flag = all(d <= LANDMARK_SIZE for d in dists)
        if self._done_flag:
            rew += self.reward_done

        # Penalty for collision between agents
        if agent.collide:
            for other_agent in world.agents:
                if other_agent is agent: continue
                dist = get_dist(agent.state.p_pos, other_agent.state.p_pos)
                dist_min = agent.size + other_agent.size
                if dist <= dist_min:
                    # print("COLLISION")
                    rew -= self.collision_pen

        return rew

    def observation(self, agent, world):
        # Observation:
        #  - Agent state: position, velocity
        #  - Other agents: [distance x, distance y, v_x, v_y]
        #  - Other agents and objects:
        #     - If in sight: [1, distance x, distance y, v_x, v_y, one_hot_color]
        #     - If not: [0, 0, 0, 0, 0, 0, 0, 0]
        #  - Landmarks:
        #     - If in sight: [1, distance x, distance y, one_hot_color]
        #     - If not: [0, 0, 0, 0, 0, 0]
        # => Full observation dim = 2 + 2 + 4 x (nb_agents - 1) + 8 x (nb_objects - 1) + 6 x (nb_landmarks)
        # All distances are divided by max_distance to be in [0, 1]
        obs = [agent.state.p_pos, agent.state.p_vel]
        for ag in world.agents:
            if ag is agent: continue
            obs.append(np.concatenate((
                (ag.state.p_pos - agent.state.p_pos) / 2.83, # Relative position normailised into [0, 1]
                ag.state.p_vel # Velocity
            )))
        for obj_i in self.obj_order_in_obs:
            obj = world.objects[obj_i]
            if get_dist(agent.state.p_pos, obj.state.p_pos) <= self.obs_range:
                obs.append(np.concatenate((
                    [1.0], # Bit saying entity is observed
                    (obj.state.p_pos - agent.state.p_pos) / self.obs_range, # Relative position normalised into [0, 1]
                    obj.state.p_vel, # Velocity
                    np.eye(3)[obj.num_color - 1] # Color
                )))
            else:
                obs.append(np.array([0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
        for lm_i in self.lm_order_in_obs:
            lm = world.landmarks[lm_i]
            if get_dist(agent.state.p_pos, lm.state.p_pos) <= self.obs_range:
                obs.append(np.concatenate((
                    [1.0], 
                    (lm.state.p_pos - agent.state.p_pos) / self.obs_range, # Relative position normailised into [0, 1]
                    np.eye(3)[lm.num_color - 1] # Color
                )))
            else:
                obs.append(np.array([0.0, 1.0, 1.0, 0.0, 0.0, 0.0]))

        return np.concatenate(obs)