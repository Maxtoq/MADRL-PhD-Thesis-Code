import numpy as np


class Parser():

    # vocab = ["Gem", "Yellow", "Green", "Purple", "Center", "North", "South", "East", "West"]

    def __init__(self):
        self.env_size = 10

        # self.vocab = [
        #     "Prey", "Center", "North", "South", "East", "West",
        #     "Gem", "Yellow", "Green", "Purple"]
        self.vocab = ["Red", "Green", "Blue", "Yellow", "Cyan", "Purple", 
                      "Center", "North", "South", "East", "West"]

        self.max_message_len = 3

    def _gen_perfect_message(self, agent_obs):
        # Get observed map
        pos = agent_obs[:2]
        col = tuple(agent_obs[2:5])

        if sum(col) == 0:
            return []

        m = [LM_COLORS[col]]

        if pos[0] <= 0.25:
            m.append("North")
        elif pos[0] >= 0.75:
            m.append("South")
        if pos[1] <= 0.25:
            m.append("West")
        elif pos[1] >= 0.75:
            m.append("East")
        if len(m) == 1:
            m.append("Center")
        
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


LM_COLORS = {
    (1, 1, 0): "Yellow", # yellow
    (0, 1, 1): "Cyan", # cyan
    (1, 0, 1): "Purple", # purple
    (1, 0, 0): "Red", # red
    (0, 1, 0): "Green", # green
    (0, 0, 1): "Blue"  # blue
}