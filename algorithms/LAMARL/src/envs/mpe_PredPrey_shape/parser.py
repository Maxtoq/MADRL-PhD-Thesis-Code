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