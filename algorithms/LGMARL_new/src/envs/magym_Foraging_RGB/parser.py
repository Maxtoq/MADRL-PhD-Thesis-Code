import numpy as np


class Parser():

    # vocab = ["Gem", "Yellow", "Green", "Purple", "Center", "North", "South", "East", "West"]

    def __init__(self, env_size, obs_range, max_gems_in_message=2, tell_yellow=False):
        self.env_size = env_size
        self.obs_range = obs_range
        self._tell_yellow = tell_yellow

        self.vocab = [
            "Prey", "Center", "North", "South", "East", "West",
            "Gem", "Yellow", "Green", "Purple"]
        # self.vocab = [
        #     "Gem", "Yellow", "Green", "Purple", "Center", "North", "South", 
        #     "East", "West"]

        self.max_gems_in_message = max_gems_in_message
        self.max_message_len = max_gems_in_message * 4

        if self.obs_range > 5:
            self.vocab.append("Close")
            self.max_message_len += max_gems_in_message

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

        # Discard agents and yellow gems
        keep_ids = np.where(
            ~(np.all(ent == ENT_COLORS["yellow"], axis=1) 
                | np.all(ent == ENT_COLORS["agent"], axis=1)))
        ent = ent[keep_ids]
        ent_pos = ent_pos[keep_ids]

        # Relative positions
        d = (np.arange(self.obs_range) - (self.obs_range // 2)) \
            / (self.env_size - 1)
        rel_pos = d[ent_pos]
        # Absolute positions
        abs_pos = pos + rel_pos

        # # Permute and sort by decreasing gem value (permutation allows that 
        # # gems aren't always communicated from North-West to South-East)
        # perm_ids = np.random.permutation(len(gem_values))
        # sorted_ids = perm_ids[np.argsort(gem_values[perm_ids])][::-1]

        # # Take only the first few and permute again to not have better gems
        # # always first in message
        # ids = np.random.permutation(
        #     sorted_ids[:min(len(sorted_ids), self.max_gems_in_message)])
        
        for g_i in range(min(ent.shape[0], self.max_gems_in_message)):
            if np.array_equal(ent[g_i], ENT_COLORS["green"]):
                color = "Green"
            elif np.array_equal(ent[g_i], ENT_COLORS["purple"]):
                color = "Purple"
            g = [color, "Gem"]

            if self.obs_range > 5:
                if max(np.abs(rel_pos[g_i] * (self.env_size - 1))) < 3:
                    g.append("Close")

            card = False
            if abs_pos[g_i][0] <= 0.25:
                g.append("North")
                card = True
            elif abs_pos[g_i][0] >= 0.75:
                g.append("South")
                card = True
            if abs_pos[g_i][1] <= 0.25:
                g.append("West")
                card = True
            elif abs_pos[g_i][1] >= 0.75:
                g.append("East")
                card = True

            if not card:
                g.append("Center")

            m.extend(g)
        
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


ENT_COLORS = {
    "yellow": [1, 1, 0],
    "green": [0, 1, 1],
    "purple": [1, 0, 1],
    "agent": [0, 0, 1]
}