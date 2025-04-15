import numpy as np
import random

from .env import OBS_RANGE


# Mother classes
class Parser:
    """ Base Parser """

    def __init__(self, n_agents=4, n_preys=2):
        self.vocab = ["Prey", "North", "South", "East", "West", "Center"]
        self.n_agents = n_agents
        self.n_preys = n_preys
        self.max_message_len = 3 * self.n_preys

    def _gen_perfect_message(self, agent_obs):
        '''
        Generate a description of an observation.
        Input:  
            obs: (np.array(float)) observation of the agent with all the 
                entities

        Output: sentence: list(str) The sentence generated
        '''
        m = []

        agent_pos = agent_obs[:2]

        # Prey messages
        start_prey = 2 + 6 * (self.n_agents - 1)
        for p_i in range(self.n_preys):
            prey_i = start_prey + p_i * 6

            if agent_obs[prey_i] == 1:
                p_obs = agent_obs[prey_i: prey_i + 6]
        
                p = ["Prey"]

                # ablsolute position
                prey_pos = agent_pos + p_obs[1:3] * OBS_RANGE # de-normalize

                card = False
                if prey_pos[1] >= 0.33:
                    p.append("North")
                    card = True
                elif prey_pos[1] <= -0.33:
                    p.append("South")
                    card = True
                if prey_pos[0] <= -0.33:
                    p.append("West")
                    card = True
                elif prey_pos[0] >= 0.33:
                    p.append("East")
                    card = True

                if not card:
                    p.append("Center")

                m.extend(p)

        return m

    def get_perfect_messages(self, obs_batch):
        """
        Generate perfect messages for given observations.
        :param obs_batch (np.ndarray): Batch of observations
        """
        out = []
        for e_i in range(obs_batch.shape[0]):
            env_out = []
            for a_i in range(obs_batch.shape[1]):
                env_out.append(self._gen_perfect_message(obs_batch[e_i, a_i]))
            out.append(env_out)
        return out


# class ColorParser(Parser):
#     """ Base class for parser for environment with colors. """  

#     colors = ["Red", "Green", "Blue"]

#     def __init__(self, scenar):
#         super(ColorParser, self).__init__(scenar)

#     # Get the color based on its one-hot array
#     def array_to_color(self, array):
#         # Get the color based on the array
#         idx = np.where(array == 1)[0]
#         return self.colors[idx]


# class ObservationParser(ColorParser):
    
#     vocab = ['Located', 'Object', 'Landmark', 'North', 'South', 'East', 'West', 'Center', 'Not', "Red", "Blue", "Green"]
    
#     def __init__(self, scenar, chance_not_sent):
#         """
#         ObservationParser, generate descriptions of the agents' observations.
#         Inputs:
#             scenar (Scenario): Scenario currently running.
#             chance_not_sent (float): Chance of generating a not sentence.
#         """
#         super(ObservationParser, self).__init__(scenar)
#         self.nb_agents = scenar.nb_agents
#         self.nb_objects = scenar.nb_objects
#         self.chance_not_sent = chance_not_sent

#     # Generate a sentence for the object
#     def object_sentence(self, obs):
#         '''
#         Will generate a sentence if the agent sees the object
#         with the position of this object and will generate another sentence
#         if this agent is pushing the object
#         Input:  
#             obs: list(float) observation link to this object
#         Output: 
#             sentence: list(str) The object sentence generated
#         '''
#         sentence = []

#         # If visible                                      
#         if  obs[0] == 1 :
#             sentence.append(self.array_to_color(obs[5:8]))
#             sentence.append("Object")
#             # North / South
#             if  obs[2] >= 0.25:
#                 sentence.append("North")
#             elif  obs[2] < -0.25:
#                 sentence.append("South")
            
#             # West / East
#             if  obs[1] >= 0.25:
#                 sentence.append("East")
#             elif  obs[1] < -0.25:
#                 sentence.append("West")            

#         return sentence

#      # Generate a sentence for the landmark
#     def landmark_sentence(self, obs):
#         '''
#         Will generate a sentence if the agent sees the landmark
#         with the position of this landmark

#         Input:  
#             obs: list(float) observation link to this object (1: visible or not, 2: position)

#         Output: sentence: list(str) The landmark sentence generated
#         '''
#         sentence = []
#         # Variable to check if we are close to the landmark
#         # = True until we check its position
#         close = True

#         # If visible
#         if  obs[0] == 1 :
#             sentence.append(self.array_to_color(obs[3:6]))
#             sentence.append("Landmark")
            
#             # North / South
#             if  obs[2] >= 0.2:
#                 sentence.append("North")
#                 close = False
#             elif  obs[2] < -0.2:
#                 sentence.append("South")
#                 close = False

#             # West / East
#             if  obs[1] >= 0.2:
#                 sentence.append("East")
#                 close = False
#             elif  obs[1] < -0.2:
#                 sentence.append("West")
#                 close = False
            
#             #If we are close to landmark
#             if close:
#                 # North / South
#                 if  obs[2] >= 0:
#                     sentence.append("North")
#                 elif  obs[2] < 0:
#                     sentence.append("South")
                    
#                 # West / East
#                 if  obs[1] >= 0:
#                     sentence.append("East")
#                 elif  obs[1] < 0:
#                     sentence.append("West")

#         return sentence

#     # Might generate a not sentence
#     def not_sentence(self, position, not_vis_obj, not_vis_land):
#         '''
#         Might Create a "Not sentence" if the agent don't see 1 
#         or more types of object

#         Input:  
#             position: (list(float)) The position of the agent
#             not_vis_obj: (list(int)) List of objects ids not visible
#             not_vis_land: (list(int)) List of landmarks ids not visible

#         Output: list(str) A sentence based on what it doesn't see
#         '''
#         sentence = []

#         #Generation of a NOT sentence ?
#         """
#         if = 1: Will generate not_sentence only for objects
#         if = 2: Will generate not_sentence only for landmarks
#         if = 3: Will generate not_sentence for both objects and landmarks
#         """
#         not_sentence = 0
#         # We don't always generate not sentence
#         if random.random() <= self.chance_not_sent:
#             not_sentence = random.randint(1,3)

#             if not_sentence in [1, 3] and len(not_vis_obj) != 0:
#                 # Object not sentence
#                 # Pick a random object
#                 obj_i = random.choice(not_vis_obj)
#                 obj = self.scenar.world
#                 sentence.extend(
#                     [obj_i.color_name, "Object", "Not"])
#                 for word in position:
#                     sentence.append(word)
#             if not_sentence in [2, 3] and len(not_vis_land) != 0:
#                 # Landmark not sentence
#                 # Pick a random object
#                 land_i = random.choice(not_vis_land)
#                 sentence.extend(
#                     [land_i.color_name, "Landmark", "Not"])
#                 for word in position:
#                     sentence.append(word)

#         return sentence

#     # Generate the full sentence for the agent
#     def parse_obs(self, obs):
#         '''
#         Generate a description of an observation.
#         Input:  
#             obs: (np.array(float)) observation of the agent with all the 
#                 entities

#         Output: sentence: list(str) The sentence generated
#         '''
#         # Sentence generated
#         sentence = []
#         # Position of the agent
#         position = []

#         # Get the position of the agent
#         sentence.extend(self.position_agent(obs[0:2]))
#         for i in range(1,len(sentence)):
#             position.append(sentence[i])
        
#         # Will calculate the place in the array of each entity
#         # There are 4 values for the main agent
#         """
#         2: position
#         2: velocity
#         """
#         place = 4

#         # Add the values of the other agents
#         """
#         1: visible or not
#         2: position
#         2: velocity
#         """
#         for _ in range(self.nb_agents - 1):
#             place += 5

#         # Objects sentence
#         """
#         1: visible or not
#         2: position
#         2: velocity
#         3: color
#         """
#         objects_sentence = []
#         not_visible_obj = []
#         for obj_i in range(self.nb_objects):
#             object_sentence = []
#             object_sentence.extend(self.object_sentence(obs[place:place+8]))
#             place += 8
#             # If we don't see the object
#             if len(object_sentence) == 0:
#                 # Get the number of the object
#                 not_visible_obj.append(obj_i)
#             # Else we append the sentence
#             else:
#                 objects_sentence.extend(object_sentence)
#         #no_objects = len(objects_sentence) == 0  
#         sentence.extend(objects_sentence)

#         # Landmarks sentence
#         """
#         1: visible or not
#         2: position
#         3: color
#         """
#         landmarks_sentence = []
#         not_visible_land = []
#         for lm_i in range(self.nb_objects):
#             landmark_sentence = []
#             landmark_sentence.extend(
#                 self.landmark_sentence(obs[place:place + 6]))
#             place += 6
#             # If we don't see the object
#             if len(landmark_sentence) == 0:
#                 # Get the number of the object
#                 not_visible_land.append(lm_i)
#             # Else we append the sentence
#             else:
#                 landmarks_sentence.extend(landmark_sentence)
#         #no_landmarks = len(landmarks_sentence) == 0
#         sentence.extend(landmarks_sentence)

#         # Not sentence
#         sentence.extend(self.not_sentence(
#             position, not_visible_obj, not_visible_land))

#         return sentence