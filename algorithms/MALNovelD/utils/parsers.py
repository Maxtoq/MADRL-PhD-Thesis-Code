import numpy as np

from abc import ABC, abstractmethod

# Mother classes
class Parser(ABC):
    """ Abstract Parser """
    @abstractmethod
    def parse_obs(self, obs):
        """
        Returns a sentence generated based on the actions of the agent
        """
        raise NotImplementedError

    def position_agent(self, obs):
        sentence = []
        # Position of the agent (at all time)
        sentence.append("Located")
        
        # North / South
        if  obs[1] >= 0.33:
            sentence.append("North")
        elif  obs[1] < -0.33:
            sentence.append("South")
        
        # West / East
        if  obs[0] >= 0.33:
            sentence.append("East")
        elif  obs[0] < -0.33:
            sentence.append("West")
        
        # Center
        if len(sentence) == 1:
            sentence.append("Center")

        return sentence
    
    def get_descriptions(self, obs_list):
        """
        Returns descriptions of all agents' observations.
        Inputs:
            obs_list (list(np.array)): List of observations, one for each agent.
        Output:
            descrs (list(list(str))): List of descriptions, one sentence for 
                each agent.
        """
        descr = [
            self.parse_obs(obs_list[a_i]) 
            for a_i in range(self.nb_agents)]
        return descr


class ColorParser(Parser):
    """ Base class for parser for environment with colors. """  

    colors = ["Red", "Bleu", "Green"]

    # # Get the color based on the number
    # def get_color(self, color):
    #     return self.colors[color]
        # # Red
        # if color == 1:
        #     color = "Red"
        # # Blue
        # elif color == 2:
        #     color = "Bleu"
        # # Green
        # elif color == 3:
        #     color = "Green"

        # return color

    # Get the color based on its one-hot array
    def array_to_color(self, array):
        # Get the color based on the array
        idx = np.where(array == 1)[0]
        return self.colors[idx]
        # color = None
        # # Red
        # if idx == 0:
        #     color = "Red"
        # # Blue
        # elif idx == 1:
        #     color = "Blue"
        # # Green
        # elif idx == 2:
        #     color = "Green"

        # return color

    # Get the color number based on its one-hot array
    # def array_to_num(self, array):
    #     # Get the color based on the array
    #     return np.where(array == 1)[0]
        # color = 0
        # # Red
        # if idx == 0:
        #     color = 1
        # # Blue
        # elif idx == 1:
        #     color = 2
        # # Green
        # elif idx == 2:
        #     color = 3

        # return color

