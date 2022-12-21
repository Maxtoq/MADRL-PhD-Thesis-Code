import numpy as np
import random

from multiagent.scenario import BaseScenario
from multiagent.core import World, Agent, Landmark, Action, Entity

from utils.parsers import Parser

BUTTON_RADIUS = 0.05
LANDMARK_RADIUS = 0.07
OBJECT_RADIUS = 0.2
OBJECT_MASS = 2.0
AGENT_RADIUS = 0.04
AGENT_MASS = 0.4

def get_dist(pos1, pos2, squared=False):
    dist = np.sum(np.square(pos1 - pos2))
    if squared:
        return dist
    else:
        return np.sqrt(dist)


class ObservationParser(Parser):
    
    vocab = [
        'Located', 
        'Object', 
        'Landmark', 
        'Button', 
        'Moving', 
        'Pushed', 
        'North', 
        'South', 
        'East', 
        'West', 
        'Center']

    def __init__(self, nb_agents, nb_objects, chance_not_sent):
        """
        ObservationParser, generate sentences for the agents.
        :param nb_agents: (int) Number of agents.
        :param nb_objects: (int) Number of objects.
        :param chance_not_sent: (float) Probability of generating a not 
            sentence.
        """
        self.nb_agents = nb_agents
        self.nb_objects = nb_objects
        self.chance_not_sent = chance_not_sent

    def object_sentence(self, obj_info):
        """
        Generates the part of the sentence concerning an object.
        Input:  
            obj_info: (numpy.ndarray(float)) Part of the observation linked to
                the object (1: visible or not, 2: position, 2: velocity).
        Output: phrase: (list(str)) The generated phrase.
        """
        phrase = []

        # If visible                                      
        if  obj_info[0] == 1.0:
            phrase.append("Object")
            # North / South
            if  obj_info[2] > AGENT_RADIUS:
                phrase.append("North")
            elif  obj_info[2] < -AGENT_RADIUS:
                phrase.append("South")
            # West / East
            if  obj_info[1] > AGENT_RADIUS:
                phrase.append("East")
            elif  obj_info[1] < -AGENT_RADIUS:
                phrase.append("West")

            moves = ["Moving"]
            if obj_info[4] > 0.05:
                moves.append("North")
            elif obj_info[4] < -0.05:
                moves.append("South")
            if obj_info[3] > 0.05:
                moves.append("East")
            elif obj_info[3] < -0.05:
                moves.append("West")
            if len(moves) > 1:
                phrase.extend(moves)

        return phrase

    def landmark_sentence(self, lm_info):
        """
        Generates the part of the sentence concerning a landmark.
        Input:  
            lm_info: (numpy.ndarray(float)) Part of the observation linked to
                the landmark (1: visible or not, 2: position).
        Output: phrase: (list(str)) The generated phrase.
        """
        phrase = []

        # If visible                                      
        if  lm_info[0] == 1.0:
            phrase.append("Landmark")
            # North / South
            if  lm_info[2] > 0:
                phrase.append("North")
            else:
                phrase.append("South")
            # West / East
            if  lm_info[1] > 0:
                phrase.append("East")
            else:
                phrase.append("West")

        return phrase

    def button_sentence(self, button_info):
        """
        Generates the part of the sentence concerning a button.
        Input:  
            button_info: (numpy.ndarray(float)) Part of the observation linked to
                the button (1: visible or not, 2: position).
        Output: phrase: (list(str)) The generated phrase.
        """
        phrase = []

        # If visible                                      
        if  button_info[0] == 1.0:
            phrase.append("Button")

            if button_info[1] == 1.0:
                phrase.append("Pushed")

            # North / South
            if  button_info[3] > 0:
                phrase.append("North")
            else:
                phrase.append("South")
            # West / East
            if  button_info[2] > 0:
                phrase.append("East")
            else:
                phrase.append("West")

        return phrase

    # # Might generate a not sentence
    # def not_sentence(self, position, no_objects, no_landmarks):
    #     '''
    #     Might Create a "Not sentence" if the agent don't see 1 
    #     or more types of object

    #     Input:  
    #         position: list(float) The position of the agent
    #         no_objects: (bool) True or False if we see an object
    #         no_landmarks: (bool) True or False if we see a landmark

    #     Output: list(str) A sentence based on what it doesn't see
    #     '''
    #     sentence = []

    #     #Generation of a NOT sentence ?
    #     """
    #     if = 1: Will generate not_sentence only for objects
    #     if = 2: Will generate not_sentence only for landmarks
    #     if = 3: Will generate not_sentence for both objects and landmarks
    #     """
    #     not_sentence = 0
    #     # We don't always generate not sentence
    #     if random.random() <= self.chance_not_sent:
    #         not_sentence = random.randint(1,3)

    #         if not_sentence == 1 and no_objects:
    #             # Object not sentence
    #             sentence.extend(["Object","Not"])
    #             for word in position:
    #                 sentence.append(word)
    #         elif not_sentence == 2 and no_landmarks:
    #             # Landmark not sentence
    #             sentence.extend(["Landmark","Not"])
    #             for word in position:
    #                 sentence.append(word)
    #         elif not_sentence == 3:
    #             # Both object
    #             if no_objects:
    #                 sentence.extend(["Object","Not"])
    #                 for word in position:
    #                     sentence.append(word)
    #             if no_landmarks:
    #                 sentence.extend(["Landmark","Not"])
    #                 for word in position:
    #                     sentence.append(word)

    #     return sentence
    
    # Generate the full sentence for the agent
    def parse_obs(self, obs):
        """
        Generate a description of the given observation.
        Input:
            obs: numpy.ndarray(float) An agent's observation of the 
                environment.
        Output: sentence: list(str) The generated sentence.
        """
        # Sentence generated
        sentence = []
        # Position of the agent
        position = []

        # Get the position of the agent
        sentence.extend(self.position_agent(obs[0:2]))
        for i in range(1,len(sentence)):
            position.append(sentence[i])

        # There are 4 values per agent
        """
        2: position
        2: velocity
        """
        place = 4 * self.nb_agents
        
        # Objects sentence
        """
        1: visible or not
        2: position
        2: velocity
        """
        obj_phrases = []
        for object in range(self.nb_objects):
            obj_phrases.extend(self.object_sentence(obs[place:place + 5]))
            place += 5
        no_objects = len(obj_phrases) == 0  
        sentence.extend(obj_phrases)

        # Landmarks sentence
        """
        1: visible or not
        2: position
        """
        lm_phrases = []
        for landmark in range(self.nb_objects):
            lm_phrases.extend(self.landmark_sentence(obs[place:place + 3]))
            place += 3
        no_landmarks = len(lm_phrases) == 0
        sentence.extend(lm_phrases)

        # Buttons sentence
        """
        1: visible or not
        1: pushed or not
        2: position
        """
        button_phrase = self.button_sentence(obs[place:place + 4])
        place += 4
        no_buttons = len(button_phrase) == 0
        sentence.extend(button_phrase)

        # # Not sentence
        # sentence.extend(self.not_sentence(position, no_objects, no_landmarks))

        return sentence


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
        # for obj_i in range(self.nb_objects):
        #     # Update dists
        #     self.obj_lm_dists[obj_i] = get_dist(
        #         self.objects[obj_i].state.p_pos,
        #         self.landmarks[obj_i].state.p_pos)
        #     # Compute reward
        #     self.global_reward += 100 * (
        #         last_obj_lm_dists[obj_i] - self.obj_lm_dists[obj_i])
        # Check if button is pushed to set movable state of objects
        objects_move = False
        for ag in self.agents:
            if self.button.is_pushing(ag.state.p_pos):
                objects_move = True
                self.global_reward += 3.0
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
                   collision_pen=15.0, reward_done=300, 
                   step_penalty=0.1, obj_lm_dist_range=[OBJECT_RADIUS + LANDMARK_RADIUS, 1.5]):
        world = ClickNPushWorld(nb_agents, nb_objects)
        # Init world entities
        self.nb_agents = nb_agents
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.silent = True
            agent.size = AGENT_RADIUS
            agent.initial_mass = AGENT_MASS
            agent.accel = 3.8
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
                obj.state.p_pos = np.array([0.0, 0.0])
                while True:
                    # obj.state.p_pos = np.array([
                    #     random.uniform(-1 + obj.size, 1 - obj.size),
                    #     random.uniform(-1 + obj.size, 1 - 2 * obj.size)])
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