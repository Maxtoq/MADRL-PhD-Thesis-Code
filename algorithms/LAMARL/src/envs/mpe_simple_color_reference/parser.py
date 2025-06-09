import numpy as np
import random

from .env import LM_COLORS


# Mother classes
class Parser:
    """ Base Parser """

    def __init__(self):
        self.vocab = ["North", "South", "East", "West", "Center", "Red", "Green", "Blue"]
        self.max_message_len = 3

    def _gen_perfect_message(self, agent_obs):
        '''
        Generate a description of an observation.
        Input:  
            obs: (np.array(float)) observation of the agent with all the 
                entities

        Output: sentence: list(str) The sentence generated
        '''
        m = []

        lm_pos = agent_obs[-5:-3]
        lm_col = agent_obs[-3:]

        for k, v in LM_COLORS.items():
            if (v == lm_col).all():
                m.append(k)
        
        card = False
        if lm_pos[1] >= 0.33:
            m.append("North")
            card = True
        elif lm_pos[1] <= -0.33:
            m.append("South")
            card = True
        if lm_pos[0] <= -0.33:
            m.append("West")
            card = True
        elif lm_pos[0] >= 0.33:
            m.append("East")
            card = True
        if not card:
            m.append("Center")

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