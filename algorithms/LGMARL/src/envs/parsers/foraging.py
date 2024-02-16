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

        # Sort by distance
        # for rel_pos in rel_gem_pos:
        #     print(rel_pos, np.abs(rel_pos * (self.env_size - 1)))
        
        # colors_seen = {
        #     "Yellow": False,
        #     "Green": False,
        #     "Purple": False}
        for abs_pos, rel_pos, gem_val in zip(
                abs_gem_pos, rel_gem_pos, gem_values):
            color = GEM_COLORS[gem_val]

            # # Stop if this color is already seen
            # if not colors_seen[color]:
            #     colors_seen[color] = True
            # else:
            #     continue

            p = [color, "Gem"]

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