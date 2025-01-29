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

        self.max_gems_in_message = max_gems_in_message
        self.max_message_len = max_gems_in_message * 4

        if self.obs_range > 5:
            self.vocab.append("Close")
            self.max_message_len += max_gems_in_message

    def _gen_perfect_message(self, agent_obs):
        m = []        
        return m

    def get_perfect_messages(self, obs):
        out = []
        for e_i in range(obs.shape[0]):
            env_out = []
            for a_i in range(obs.shape[1]):
                env_out.append(self._gen_perfect_message(obs[e_i, a_i]))
            out.append(env_out)
        return out
