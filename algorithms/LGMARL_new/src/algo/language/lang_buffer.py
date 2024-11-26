import random
import numpy as np


class LanguageBuffer:

    def __init__(self, 
            buffer_size, n_agents, obs_dim, joint_obs_dim, hidden_dim, 
            recurrent_N, max_message_len, batch_size, n_rollout_steps):
        self.buffer_size = buffer_size
        self.n_agents = n_agents
        self.recurrent_N = recurrent_N
        self.batch_size = batch_size
        self.n_rollout_steps = min(buffer_size, n_rollout_steps)

        # Decoder data
        self.obs = np.zeros(
            (self.buffer_size, self.n_agents, obs_dim),
            dtype=np.float32)
        self.obs_enc_rnn_states = np.zeros(
            (self.buffer_size, self.n_agents, self.recurrent_N, hidden_dim),
            dtype=np.float32)
        self.perf_messages = np.zeros(
            (self.buffer_size, self.n_agents, max_message_len),
            dtype=np.int8)

        # Encoder data
        self.joint_obs = np.zeros(
            (self.buffer_size, joint_obs_dim),
            dtype=np.float32)
        self.joint_obs_enc_rnn_states = np.zeros(
            (self.buffer_size, self.n_agents, self.recurrent_N, hidden_dim),
            dtype=np.float32)
        self.perf_broadcasts = []

        self._current_id = 0

    def roll_n(self, n_index):
        self.obs = np.roll(self.obs, -n_index, axis=0)
        self.obs_enc_rnn_states = np.roll(
            self.obs_enc_rnn_states, -n_index, axis=0)
        self.perf_messages = np.roll(self.perf_messages, -n_index, axis=0)
        self.joint_obs = np.roll(self.joint_obs, -n_index, axis=0)
        self.joint_obs_enc_rnn_states = np.roll(
            self.joint_obs_enc_rnn_states, -n_index, axis=0)
        self.perf_broadcasts = self.perf_broadcasts[n_index:]

        add_size = n_index + self.buffer_size - self._current_id
        self._current_id = self.buffer_size - add_size

    def store(self, 
            obs, joint_obs, perf_messages, perf_broadcasts, 
            obs_enc_rnn_states, joint_obs_enc_rnn_states):
        """
        Store language data in buffer.
        :param obs: (np.ndarray) local observations, shape=(n_envs, n_agents, 
            obs_dim)
        :param joint_obs: (np.ndarray) joint observations, shape=(n_envs, 
            n_agents, joint_obs_dim)
        :param perf_messages: (np.ndarray) perfect messages, shape=(n_envs, 
            n_agents, max_message_len)
        :param perf_broadcasts: (list(list(int))) perfect broadcasts, in list
            because they're not padded (better for encoding), shape=(n_envs, br_len)
        """        
        add_size = obs.shape[0]

        # Slide data if needed 
        if self._current_id + add_size > self.buffer_size:
            self.roll_n(self.n_rollout_steps) # fits when buffer_size is divisible by n_rollout_steps, overwise not perfect (some data may be lost too early) but ok
    
        # Store data
        self.obs[self._current_id: self._current_id + add_size] = obs
        self.obs_enc_rnn_states[self._current_id: self._current_id + add_size] \
            = obs_enc_rnn_states
        self.perf_messages[
            self._current_id: self._current_id + add_size] = perf_messages
        self.joint_obs[self._current_id: self._current_id + add_size] \
            = joint_obs[:, 0]
        self.joint_obs_enc_rnn_states[
            self._current_id: self._current_id + add_size] \
            = joint_obs_enc_rnn_states
        self.perf_broadcasts += perf_broadcasts

        self._current_id += add_size

    def sample(self):
        batch_size = min(self.batch_size, self._current_id)

        ids = np.random.choice(
            self._current_id, size=batch_size, replace=False)

        obs_b = self.obs[ids]
        obs_rnn_state_b = self.obs_enc_rnn_states[ids]
        perf_message_b = self.perf_messages[ids]
        joint_obs_b = self.joint_obs[ids]
        joint_obs_rnn_state_b = self.joint_obs_enc_rnn_states[ids]
        perf_br_b = [self.perf_broadcasts[i] for i in ids]

        return obs_b, obs_rnn_state_b, perf_message_b, joint_obs_b, \
                joint_obs_rnn_state_b, perf_br_b