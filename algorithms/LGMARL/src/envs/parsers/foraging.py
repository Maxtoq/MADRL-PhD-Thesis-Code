import numpy as np


GEM_COLORS = {
    1: 'Yellow',
    2: 'Green',
    3: 'Purple'
}


class Foraging_Parser():

    # vocab = ["Gem", "Yellow", "Green", "Purple", "Center", "North", "South", "East", "West"]

    def __init__(self, env_size, obs_range):
        self.env_size = env_size
        self.obs_range = obs_range

        self.vocab = [
            "Gem", "Yellow", "Green", "Purple", "Center", "North", "South", 
            "East", "West"]

        if self.obs_range > 5:
            self.vocab.append("Close")

    # def _get_pos_sent(self, pos):
    #     """
    #     Construct part of sentence describing the position of an agent.
    #     :param pos: (list(float)) 2D position.
    #     :return sent: (list(str)) Language position.
    #     """
    #     sent = ["Located"]
    #     if pos[0] <= 0.25:
    #         sent.append("North")
    #     elif pos[0] >= 0.75:
    #         sent.append("South")
    #     if pos[1] <= 0.25:
    #         sent.append("West")
    #     elif pos[1] >= 0.75:
    #         sent.append("East")

    #     if len(sent) == 1:
    #         sent.append("Center")

    #     return sent

    # def _get_gem_sent(self, gems):
    #     """
    #     Construct part of sentence describing the position of an agent.
    #     :param gems: (list(float)) List of surrounding tiles.
    #     :return sent: (list(str)) Sentence describing observed gems.
    #     """
    #     if 1.0 in gems:
    #         gem_map = np.array(gems).reshape((5, 5))
    #         gem_pos = [(x, y) 
    #             for x in range(5) 
    #                 for y in range(5) 
    #                     if gem_map[y, x] == 1.0]
    #         sent = []
    #         for p in gem_pos:
    #             sent.append("Prey")
    #             sent.append("Observed")
    #             if p[1] < 2:
    #                 sent.append("North")
    #             elif p[1] > 2:
    #                 sent.append("South")
    #             if p[0] < 2:
    #                 sent.append("West")
    #             elif p[0] > 2:
    #                 sent.append("East")

    #         return sent
    #     else:
    #         return []

    # def parse_observations(self, obs):
    #     """
    #     Parse local observations.
    #     :param obs: (list(list(float))) List of observations.
    #     :return sentences: (list(list(str))) List of sentences.
    #     """
    #     sentences = []
    #     for o in obs:
    #         s = []
    #         pos = o[:2]
    #         gems = o[2:]

    #         # Get position part of sentence
    #         s += self._get_pos_sent(pos)

    #         # Get gem part of the observation
    #         s += self._get_gem_sent(gems)

    #         sentences.append(s)
    #     return sentences

    def _gen_perfect_message(self, agent_obs):
        m = []
        pos = agent_obs[:2]
        gem_map = np.array(
            agent_obs[2:]).reshape((self.obs_range, self.obs_range))

        d = (np.arange(self.obs_range) - (self.obs_range // 2)) \
                / (self.env_size - 1)
        rel_gem_pos = np.stack([d[ax] for ax in np.nonzero(gem_map)]).T
        gem_values = gem_map[np.nonzero(gem_map)]
        abs_gem_pos = pos + rel_gem_pos

        for abs_pos, rel_pos, gem_val in zip(
                abs_gem_pos, rel_gem_pos, gem_values):
            p = [GEM_COLORS[gem_val], "Gem"]

            if self.obs_range > 5:
                if max(np.abs(rel_pos * (self.env_size - 1))) < 3:
                    p.append("Close")

            if abs_pos[0] <= 0.25:
                p.append("North")
            elif abs_pos[0] >= 0.75:
                p.append("South")
            if abs_pos[1] <= 0.25:
                p.append("West")
            elif abs_pos[1] >= 0.75:
                p.append("East")

            if len(p) == 1:
                p.append("Center")

            m.extend(p)
        
        return m

    def get_perfect_messages(self, obs):
        """
        Recurrent method for generating perfect messages corresponding to
        given observations.
        :param obs (np.ndarray): Batch of observations
        """
        out = []
        for e_i in range(obs.shape[0]):
            env_out = []
            for a_i in range(obs.shape[1]):
                env_out.append(self._gen_perfect_message(obs[e_i, a_i]))
            out.append(env_out)
        return out