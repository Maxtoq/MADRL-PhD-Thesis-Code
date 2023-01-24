from .buffer import ReplayBuffer, RecReplayBuffer


class PrioritizedRecReplayBuffer(RecReplayBuffer):
    def __init__(self, 
            alpha, buffer_size, episode_length, nb_agents, obs_dim, act_dim):
        """ Prioritized replay buffer class for training RNN policies. """
        super(PrioritizedRecReplayBuffer, self).__init__(
            buffer_size, episode_length, nb_agents, obs_dim, act_dim)
        self.alpha = alpha
        it_capacity = 1
        while it_capacity < buffer_size:
            it_capacity *= 2

        self._it_sums = {
            a_id: SumSegmentTree(it_capacity) 
            for a_id in range(nb_agents)}
        self._it_mins = {
            a_id: MinSegmentTree(it_capacity) 
            for a_id in range(nb_agents)}
        self.max_priorities = {a_id: 1.0 for a_id in self.policy_info.keys()}

    def insert(self, num_insert_episodes, obs, share_obs, acts, rewards, dones, dones_env, avail_acts=None):
        """See parent class."""
        idx_range = super().insert(num_insert_episodes, obs, share_obs, acts, rewards, dones, dones_env, avail_acts)
        for idx in range(idx_range[0], idx_range[1]):
            for p_id in self.policy_info.keys():
                self._it_sums[p_id][idx] = self.max_priorities[p_id] ** self.alpha
                self._it_mins[p_id][idx] = self.max_priorities[p_id] ** self.alpha

        return idx_range

    def _sample_proportional(self, batch_size, p_id=None):
        total = self._it_sums[p_id].sum(0, len(self) - 1)
        mass = np.random.random(size=batch_size) * total
        idx = self._it_sums[p_id].find_prefixsum_idx(mass)
        return idx

    def sample(self, batch_size, beta=0, p_id=None):
        """
        Sample a set of episodes from buffer; probability of choosing a given episode is proportional to its priority.
        :param batch_size: (int) number of episodes to sample.
        :param beta: (float) controls the amount of prioritization to apply.
        :param p_id: (str) policy which will be updated using the samples.

        :return: See parent class.
        """
        assert len(
            self) > batch_size, "Cannot sample with no completed episodes in the buffer!"
        assert beta > 0

        batch_inds = self._sample_proportional(batch_size, p_id)

        p_min = self._it_mins[p_id].min() / self._it_sums[p_id].sum()
        max_weight = (p_min * len(self)) ** (-beta)
        p_sample = self._it_sums[p_id][batch_inds] / self._it_sums[p_id].sum()
        weights = (p_sample * len(self)) ** (-beta) / max_weight

        obs, share_obs, acts, rewards, dones, dones_env, avail_acts = {}, {}, {}, {}, {}, {}, {}
        for p_id in self.policy_info.keys():
            p_buffer = self.policy_buffers[p_id]
            obs[p_id], share_obs[p_id], acts[p_id], rewards[p_id], dones[p_id], dones_env[p_id], avail_acts[
                p_id] = p_buffer.sample_inds(batch_inds)

        return obs, share_obs, acts, rewards, dones, dones_env, avail_acts, weights, batch_inds

    def update_priorities(self, idxes, priorities, p_id=None):
        """
        Update priorities of sampled transitions.
        sets priority of transition at index idxes[i] in buffer
        to priorities[i].
        :param idxes: ([int]) List of idxes of sampled transitions
        :param priorities: ([float]) List of updated priorities corresponding to transitions at the sampled idxes
            denoted by variable `idxes`.
        """
        assert len(idxes) == len(priorities)
        assert np.min(priorities) > 0
        assert np.min(idxes) >= 0
        assert np.max(idxes) < len(self)

        self._it_sums[p_id][idxes] = priorities ** self.alpha
        self._it_mins[p_id][idxes] = priorities ** self.alpha

        self.max_priorities[p_id] = max(
            self.max_priorities[p_id], np.max(priorities))