import random
from math import sqrt

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
            return "Red"
        # Blue
        elif color == 2:
            return "Bleu"
        # Green
        elif color == 3:
            return "Green"
        # Yellow
        elif color == 4:
            return "Yellow"
        # Purple
        elif color == 5:
            return "Purple"
        #Black
        elif color == 6:
            return "Black"
        else:
            return None
        
        
class ColorShapeParser(ColorParser):
    """ Abstract Parser """
    # Get the shape based on the number
    def get_shape(self, shape):
        #Black
        if shape == 1:
            return "Circle"
        # Red
        elif shape == 2:
            return "Square"
        # Blue
        elif shape == 3:
            return "Triangle"
        else:
            return None

