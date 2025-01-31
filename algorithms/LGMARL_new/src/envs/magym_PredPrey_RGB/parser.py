import numpy as np


class Parser():

    # vocab = ["Prey", "Located", "Observed", "Center", "North", "South", "East", "West"]

    def __init__(self, env_size, obs_range, n_preys):
        self.env_size = env_size
        self.obs_range = obs_range

        self.obs_dim = 2 + obs_range * obs_range

        self.vocab = ["Prey", "Center", "North", "South", "East", "West",
                        "Gem", "Yellow", "Green", "Purple"]
        # self.vocab = ["Prey", "Center", "North", "South", "East", "West"]
        self.max_n_prey = min(2, n_preys) # Max number of preys that will be communicated about in one message
        self.max_message_len = 3 * self.max_n_prey

        if self.obs_range > 5:
            self.vocab.append("Close")
            self.max_message_len += self.max_n_prey

    # def parse_global_state(self, state):
    #     """
    #     Parse the global state of the environment to produce a complete textual
    #     description of it.
    #     :param state: (np.ndarray) 2D map of the environment.
    #     :return descr: (list(list(str))) List of sentences describing the
    #         environment.
    #     """
    #     pass

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

    # def _get_prey_sent(self, preys):
    #     """
    #     Construct part of sentence describing the position of an agent.
    #     :param preys: (list(float)) List of surrounding tiles.
    #     :return sent: (list(str)) Sentence describing observed preys.
    #     """
    #     if 1.0 in preys:
    #         prey_map = np.array(preys).reshape((5, 5))
    #         prey_pos = [(x, y) 
    #             for x in range(5) 
    #                 for y in range(5) 
    #                     if prey_map[y, x] == 1.0]
    #         sent = []
    #         for p in prey_pos:
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
    #         preys = o[2:]

    #         # Get position part of sentence
    #         s += self._get_pos_sent(pos)

    #         # Get prey part of the observation
    #         s += self._get_prey_sent(preys)

    #         sentences.append(s)
    #     return sentences

    def _gen_perfect_message(self, agent_obs):
        m = []

        # Get observed map
        pos = agent_obs[:2]
        obs_map = np.array(
            agent_obs[2:]).reshape((self.obs_range, self.obs_range, 3))

        # Observed entities positions
        ent_pos = np.transpose(np.nonzero(obs_map.sum(-1) > 0))

        # Entities
        ent = obs_map[ent_pos[:, 0], ent_pos[:, 1], :]
        # Relative positions
        d = (np.arange(self.obs_range) - (self.obs_range // 2)) \
            / (self.env_size - 1)
        rel_pos = d[ent_pos]
        # Absolute positions
        abs_pos = pos + rel_pos

        for p_i in range(min(ent.shape[0], self.max_n_prey)):
            if not np.array_equal(ent[p_i], [1, 0, 0]):
                continue

            p = ["Prey"]

            if self.obs_range > 5:
                if max(np.abs(rel_pos[p_i] * (self.env_size - 1))) < 3:
                    p.append("Close")

            card = False
            if abs_pos[p_i][0] <= 0.25:
                p.append("North")
                card = True
            elif abs_pos[p_i][0] >= 0.75:
                p.append("South")
                card = True
            if abs_pos[p_i][1] <= 0.25:
                p.append("West")
                card = True
            elif abs_pos[p_i][1] >= 0.75:
                p.append("East")
                card = True

            if not card:
                p.append("Center")

            m.extend(p)
        
        return m

    def get_perfect_messages(self, obs_batch):
        """
        Recurrent method for generating perfect messages corresponding to
        given observations.
        :param obs_batch (np.ndarray): Batch of observations
        """
        out = []
        for e_i in range(obs_batch.shape[0]):
            env_out = []
            for a_i in range(obs_batch.shape[1]):
                env_out.append(self._gen_perfect_message(obs_batch[e_i, a_i]))
            out.append(env_out)
        return out
    
    def _gen_prey_pos(self, prey_pos):
        p = ["Prey"]

        card = False
        if prey_pos[0] < 6:
            p.append("North")
            card = True
        elif prey_pos[0] >= 12:
            p.append("South")
            card = True
        if prey_pos[1] < 6:
            p.append("West")
            card = True
        elif prey_pos[1] >= 12:
            p.append("East")
            card = True

        if not card:
            p.append("Center")

        return p
    
    def get_prey_pos(self, prey_pos):
        out = []
        for pp in prey_pos:
            env_out = []
            for p in pp.values():
                env_out.extend(self._gen_prey_pos(p))
            out.append([env_out])
        return out

    def check_obs(self, obs, sentence):
        if len(obs) > self.obs_dim:
            obs = obs[:self.obs_dim]
        parsed = self._gen_perfect_message(obs)
        return parsed == sentence