import numpy as np


GEM_COLORS = {
    1: 'Yellow',
    2: 'Green',
    3: 'Purple'}


class Parser():

    # vocab = ["Gem", "Yellow", "Green", "Purple", "Center", "North", "South", "East", "West"]

    def __init__(self, env_size, obs_range, max_gems_in_message=2):
        self.env_size = env_size
        self.obs_range = obs_range

        self.vocab = [
            "Gem", "Yellow", "Green", "Purple", "Center", "North", "South", 
            "East", "West"]

        self.max_gems_in_message = max_gems_in_message
        self.max_message_len = max_gems_in_message * 4

        if self.obs_range > 5:
            self.vocab.append("Close")
            self.max_message_len += max_gems_in_message

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

        # Permute and sort by decreasing gem value (permutation allows that 
        # gems aren't always communicated from North-West to South-East)
        perm_ids = np.random.permutation(len(gem_values))
        sorted_ids = perm_ids[np.argsort(gem_values[perm_ids])][::-1]

        # Take only the first few and permute again to not have better gems 
        # always first in message
        ids = np.random.permutation(
            sorted_ids[:min(len(sorted_ids), self.max_gems_in_message)])
        
        for g_i in ids:
            color = GEM_COLORS[gem_values[g_i]]
            g = [color, "Gem"]

            if self.obs_range > 5:
                if max(np.abs(rel_gem_pos[g_i] * (self.env_size - 1))) < 3:
                    g.append("Close")

            if abs_gem_pos[g_i][0] <= 0.25:
                g.append("North")
            elif abs_gem_pos[g_i][0] >= 0.75:
                g.append("South")
            if abs_gem_pos[g_i][1] <= 0.25:
                g.append("West")
            elif abs_gem_pos[g_i][1] >= 0.75:
                g.append("East")

            if len(g) == 2:
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