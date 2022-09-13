import numpy as np

from abc import ABC, abstractmethod

# Mother classes
class Parser(ABC):
    """ Abstract Parser """
    @abstractmethod
    def parse_obs(self, obs, sce_conf):
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
        if  obs[1] < -0.33:
            sentence.append("South")
        
        # West / East
        if  obs[0] >= 0.33:
            sentence.append("East")
        if  obs[0] < -0.33:
            sentence.append("West")
        
        # Center
        elif len(sentence) == 1:
            sentence.append("Center")

        return sentence

    """@abstractmethod
    def objects_sentence(self, obs, sce_conf):
        
        Returns a sentence generated based on the objects see or not by the agent
        
        raise NotImplementedError"""

    @abstractmethod
    def landmarks_sentence(self, obs, sce_conf):
        """
        Returns a sentence generated based on the landmarks see or not by the agent
        """
        raise NotImplementedError

    @abstractmethod
    def reset(self, sce_conf, colors, shapes):
        """
        Returns a sentence generated based on the landmarks see or not by the agent
        """
        raise NotImplementedError

class ColorParser(Parser):
    """ Abstract Parser """                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
    # Get the color based on the number
    def get_color(self, color):
        # Red
        if color == 1:
            color = "Red"
        # Blue
        elif color == 2:
            color = "Bleu"
        # Green
        elif color == 3:
            color = "Green"

        return color

    # Get the color based on its array
    def array_to_color(self, array):
        # Get the color based on the array
        idx = np.where(array == 1)[0]
        color = None
        # Red
        if idx == 0:
            color = "Red"
        # Blue
        elif idx == 1:
            color = "Blue"
        # Green
        elif idx == 2:
            color = "Green"

        return color

    # Get the color number based on its array
    def array_to_num(self, array):
        # Get the color based on the array
        idx = np.where(array == 1)[0]
        color = 0
        # Red
        if idx == 0:
            color = 1
        # Blue
        elif idx == 1:
            color = 2
        # Green
        elif idx == 2:
            color = 3

        return color

