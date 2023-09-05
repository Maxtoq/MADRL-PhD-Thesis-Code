import numpy as np


class PredatorPrey_Parser():

    vocab = ["Prey", "Located", "Observed", "Center", "North", "South", "East", "West"]

    def __init__(self):
        # self.obs_range = obs_range
        pass

    def parse_global_state(self, state):
        """
        Parse the global state of the environment to produce a complete textual
        description of it.
        :param state: (np.ndarray) 2D map of the environment.
        :return descr: (list(list(str))) List of sentences describing the
            environment.
        """
        pass
        # Est-ce qu'on a vraiment besoin de la description globale ?
        # Commencer par le parser d'observations locales

    def _get_pos_sent(self, pos):
        """
        Construct part of sentence describing the position of an agent.
        :param pos: (list(float)) 2D position.
        :return sent: (list(str)) Language position.
        """
        sent = ["Located"]
        if pos[0] <= 0.25:
            sent.append("North")
        elif pos[0] >= 0.75:
            sent.append("South")
        if pos[1] <= 0.25:
            sent.append("West")
        elif pos[1] >= 0.75:
            sent.append("East")

        if len(sent) == 1:
            sent.append("Center")

        return sent

    def _get_prey_sent(self, preys):
        """
        Construct part of sentence describing the position of an agent.
        :param preys: (list(float)) List of surrounding tiles.
        :return sent: (list(str)) Sentence describing observed preys.
        """
        if 1.0 in preys:
            prey_map = np.array(preys).reshape((5, 5))
            prey_pos = [(x, y) 
                for x in range(5) 
                    for y in range(5) 
                        if prey_map[y, x] == 1.0]
            sent = []
            for p in prey_pos:
                sent.append("Prey")
                sent.append("Observed")
                if p[1] < 2:
                    sent.append("North")
                elif p[1] > 2:
                    sent.append("South")
                if p[0] < 2:
                    sent.append("West")
                elif p[0] > 2:
                    sent.append("East")

            return sent
        else:
            return []

    def parse_observations(self, obs):
        """
        Parse local observations.
        :param obs: (list(list(float))) List of observations.
        :return sentences: (list(list(str))) List of sentences.
        """
        sentences = []
        for o in obs:
            s = []
            pos = o[:2]
            preys = o[2:]

            # Get position part of sentence
            s += self._get_pos_sent(pos)

            # Get prey part of the observation
            s += self._get_prey_sent(preys)

            sentences.append(s)
        return sentences