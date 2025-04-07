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
        """ Generate sentence based on prey coordinates. """
        out = []
        for pp in prey_pos:
            env_out = []
            for p in pp.values():
                env_out.extend(self._gen_prey_pos(p))
            out.append([env_out])
        return out

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

    def check_obs(self, obs, sentence):
        if len(obs) > self.obs_dim:
            obs = obs[:self.obs_dim]
        parsed = self._gen_perfect_message(obs)
        return parsed == sentence