import torch
import numpy as np


def _flatten(T, N, x):
    return x.reshape(T * N, *x.shape[2:])

def _cast(x):
    return x.transpose(1, 2, 0, 3).reshape(-1, *x.shape[3:])


class ACC_ReplayBuffer:

    def __init__(self, 
            args, n_agents, policy_input_dim, critic_input_dim, env_act_dim, comm_act_dim,
            obs_dim):
        self.n_agents = n_agents
        self.policy_input_dim = policy_input_dim
        self.critic_input_dim = critic_input_dim
        self.env_act_dim = env_act_dim
        self.comm_act_dim = comm_act_dim
        self.obs_dim = obs_dim

        self.episode_length = args.episode_length
        self.n_parallel_envs = args.n_parallel_envs
        self.hidden_size = args.hidden_dim
        self.recurrent_N = args.policy_recurrent_N
        self.gamma = args.gamma
        self.gae_lambda = args.gae_lambda
        self.n_mini_batch = args.n_mini_batch
        self.data_chunk_length = args.data_chunk_length
        self.share_params = args.share_params

        self.policy_input = np.zeros(
            (self.episode_length + 1, 
             self.n_parallel_envs, 
             self.n_agents, 
             self.policy_input_dim),
            dtype=np.float32)
        self.critic_input = np.zeros(
            (self.episode_length + 1, 
             self.n_parallel_envs, 
             self.n_agents,
             self.critic_input_dim),
            dtype=np.float32)

        self.rnn_states = np.zeros(
            (self.episode_length + 1, 
             self.n_parallel_envs, 
             self.n_agents,
             self.recurrent_N, 
             self.hidden_size), 
            dtype=np.float32)
        self.critic_rnn_states = np.zeros_like(self.rnn_states)

        self.value_preds = np.zeros(
            (self.episode_length + 1, self.n_parallel_envs, self.n_agents, 1),
            dtype=np.float32)
        self.returns = np.zeros(
            (self.episode_length + 1, self.n_parallel_envs, self.n_agents, 1),
            dtype=np.float32)

        self.env_actions = np.zeros(
            (self.episode_length, 
             self.n_parallel_envs, 
             self.n_agents, 
             self.env_act_dim),
            dtype=np.float32)
        self.env_action_log_probs = np.zeros(
            (self.episode_length, 
             self.n_parallel_envs, 
             self.n_agents, 
             self.env_act_dim),
            dtype=np.float32)

        self.comm_actions = np.zeros(
            (self.episode_length, 
             self.n_parallel_envs, 
             self.n_agents, 
             self.comm_act_dim),
            dtype=np.float32)
        self.comm_action_log_probs = np.zeros(
            (self.episode_length, 
             self.n_parallel_envs, 
             self.n_agents, 
             self.comm_act_dim),
            dtype=np.float32)

        self.rewards = np.zeros(
            (self.episode_length, self.n_parallel_envs, self.n_agents, 1), 
            dtype=np.float32)
        
        self.masks = np.ones(
            (self.episode_length + 1, self.n_parallel_envs, self.n_agents, 1), 
            dtype=np.float32)

        # Language data
        self.obs = np.zeros(
            (self.episode_length + 1, 
             self.n_parallel_envs, 
             self.n_agents, 
             self.obs_dim),
            dtype=np.float32)
        self.parsed_obs = []

        self.step = 0
    
    def reset_episode(self):
        self.policy_input = np.zeros((self.episode_length + 1, self.n_parallel_envs, self.n_agents, self.policy_input_dim), dtype=np.float32)
        self.critic_input = np.zeros((self.episode_length + 1, self.n_parallel_envs, self.n_agents, self.critic_input_dim), dtype=np.float32)
        self.rnn_states = np.zeros((self.episode_length + 1, self.n_parallel_envs, self.n_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        self.critic_rnn_states = np.zeros_like(self.rnn_states)
        self.value_preds = np.zeros((self.episode_length + 1, self.n_parallel_envs, self.n_agents, 1), dtype=np.float32)
        self.returns = np.zeros((self.episode_length + 1, self.n_parallel_envs, self.n_agents, 1), dtype=np.float32)
        self.env_actions = np.zeros((self.episode_length, self.n_parallel_envs, self.n_agents, self.env_act_dim), dtype=np.float32)
        self.env_action_log_probs = np.zeros((self.episode_length, self.n_parallel_envs, self.n_agents, self.env_act_dim), dtype=np.float32)
        self.comm_actions = np.zeros((self.episode_length, self.n_parallel_envs, self.n_agents, self.comm_act_dim), dtype=np.float32)
        self.comm_action_log_probs = np.zeros((self.episode_length, self.n_parallel_envs, self.n_agents, self.comm_act_dim), dtype=np.float32)
        self.rewards = np.zeros((self.episode_length, self.n_parallel_envs, self.n_agents, 1), dtype=np.float32)    
        self.masks = np.ones((self.episode_length + 1, self.n_parallel_envs, self.n_agents, 1), dtype=np.float32)
        self.obs = np.zeros((self.episode_length + 1, self.n_parallel_envs, self.n_agents, self.obs_dim), dtype=np.float32)
        self.parsed_obs = []
        self.step = 0

    def start_new_episode(self):
        self.policy_input[0] = self.policy_input[-1].copy()
        self.critic_input[0] = self.critic_input[-1].copy()
        self.rnn_states[0] = self.rnn_states[-1].copy()
        self.critic_rnn_states[0] = self.critic_rnn_states[-1].copy()
        self.masks[0] = self.masks[-1].copy()
        self.obs[0] = self.obs[-1].copy()
        self.parsed_obs = [self.parsed_obs[-1]]
        self.step = 0

    def get_act_params(self):
        return self.policy_input[self.step], self.critic_input[self.step], \
               self.rnn_states[self.step], self.critic_rnn_states[self.step], \
               self.masks[self.step]

    def insert_obs(self, policy_input, critic_input, obs, parsed_obs):
        self.policy_input[self.step] = policy_input
        self.critic_input[self.step] = critic_input
        self.obs[self.step] = obs
        self.parsed_obs.append(parsed_obs)

    def insert_act(self, 
            rnn_states, critic_rnn_states, env_actions, env_action_log_probs, 
            comm_actions, comm_action_log_probs, value_preds, rewards, masks):
        self.rnn_states[self.step + 1] = rnn_states.copy()
        self.critic_rnn_states[self.step + 1] = critic_rnn_states.copy()
        self.env_actions[self.step] = env_actions.copy()
        self.env_action_log_probs[self.step] = env_action_log_probs.copy()
        self.comm_actions[self.step] = comm_actions.copy()
        self.comm_action_log_probs[self.step] = comm_action_log_probs.copy()
        self.value_preds[self.step] = value_preds.copy()
        self.rewards[self.step] = rewards.copy()
        self.masks[self.step + 1] = masks.copy()
        self.step += 1

    def compute_returns(self, next_value, value_normalizer):
        """
        Compute returns either as discounted sum of rewards, or using GAE.
        :param next_value: (np.ndarray) value predictions for the step after 
            the last episode step.
        :param value_normalizer: (ValueNorm) Value normalizer instance.
        """
        self.value_preds[-1] = next_value
        gae = 0
        for step in reversed(range(self.rewards.shape[0])):
            delta = self.rewards[step] + self.gamma \
                * value_normalizer.denormalize(
                    self.value_preds[step + 1]) * self.masks[step + 1] \
                    - value_normalizer.denormalize(self.value_preds[step])
            gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] \
                * gae
            self.returns[step] = gae + value_normalizer.denormalize(
                self.value_preds[step])

    def sample_clip(self):
        """
        Returns samples for clip training.
        """
        all_obs = self.obs.reshape(
            (self.episode_length + 1) * self.n_parallel_envs * self.n_agents, -1)
        all_parsed_obs = [
            env_sentences[a_i]
            for step_sentences in self.parsed_obs
            for env_sentences in step_sentences
            for a_i in range(self.n_agents)]

        return all_obs, all_parsed_obs

    def sample_capt(self):
        """
        Returns all buffered data for captioning training.
        """
        if self.share_params:
            policy_input_batch = self.policy_input.reshape(
                (self.episode_length + 1) * self.n_parallel_envs \
                    * self.n_agents, -1)
            masks_batch = self.masks.reshape(
                (self.episode_length + 1) * self.n_parallel_envs \
                    * self.n_agents, -1)
            rnn_states_batch = self.rnn_states[0].reshape(
                self.n_parallel_envs * self.n_agents, self.recurrent_N, -1)
            parsed_obs_batch = [
                env_sentences[a_i]
                for step_sentences in self.parsed_obs
                for env_sentences in step_sentences
                for a_i in range(self.n_agents)]
        else:
            policy_input_batch = self.policy_input.reshape(
                (self.episode_length + 1) * self.n_parallel_envs, 
                 self.n_agents, -1)
            masks_batch = self.masks.reshape(
                (self.episode_length + 1) * self.n_parallel_envs,
                 self.n_agents, -1)
            rnn_states_batch = self.rnn_states[0]
            
            parsed_obs_batch = [
                [env_sentences[a_i]
                 for step_sentences in self.parsed_obs
                 for env_sentences in step_sentences]
                for a_i in range(self.n_agents)]

        return policy_input_batch, masks_batch, rnn_states_batch, parsed_obs_batch
            

    def recurrent_policy_generator(self, advantages):
        """
        Generates sample for policy training.
        """
        # Shuffled env ids
        env_ids = np.random.choice(
            self.n_parallel_envs, size=self.n_parallel_envs, replace=False)
        
        # Cut env ids into n_mini_batch parts
        mini_batch_size = self.n_parallel_envs // self.n_mini_batch # mini_batch_size = n episodes in mini batch
        if mini_batch_size == 0:
            mini_batch_size = 1
            self.n_mini_batch = 1
        sample_ids = [
            env_ids[i * mini_batch_size:(i + 1) * mini_batch_size] 
            for i in range(self.n_mini_batch)]

        for ids in sample_ids:
            policy_input_batch = self.policy_input[:-1, ids] # T x mini_batch_size x N_a x policy_input_dim
            critic_input_batch = self.critic_input[:-1, ids]
            rnn_states_batch = self.rnn_states[0, ids]
            critic_rnn_states_batch = self.critic_rnn_states[0, ids]
            env_actions_batch = self.env_actions[:, ids]
            comm_actions_batch = self.comm_actions[:, ids]
            env_action_log_probs_batch = self.env_action_log_probs[:, ids]
            comm_action_log_probs_batch = self.comm_action_log_probs[:, ids]
            value_preds_batch = self.value_preds[:-1, ids]
            returns_batch = self.returns[:-1, ids]
            masks_batch = self.masks[:-1, ids]
            advantages_batch = advantages[:, ids]

            # obs_batch = self.obs[:-1, ids]
            # parsed_obs_batch = [
            #     step_sentences[i] 
            #     for step_sentences in self.parsed_obs[:-1]
            #     for i in ids]

            if self.share_params:
                policy_input_batch = policy_input_batch.reshape(
                    self.episode_length * mini_batch_size * self.n_agents, -1)
                critic_input_batch = critic_input_batch.reshape(
                    self.episode_length * mini_batch_size * self.n_agents, -1)
                env_actions_batch = env_actions_batch.reshape(
                    self.episode_length * mini_batch_size * self.n_agents, -1)
                comm_actions_batch = comm_actions_batch.reshape(
                    self.episode_length * mini_batch_size * self.n_agents, -1)
                env_action_log_probs_batch = env_action_log_probs_batch.reshape(
                    self.episode_length * mini_batch_size * self.n_agents, -1)
                comm_action_log_probs_batch = comm_action_log_probs_batch.reshape(
                    self.episode_length * mini_batch_size * self.n_agents, -1)
                value_preds_batch = value_preds_batch.reshape(
                    self.episode_length * mini_batch_size * self.n_agents, -1)
                returns_batch = returns_batch.reshape(
                    self.episode_length * mini_batch_size * self.n_agents, -1)
                masks_batch = masks_batch.reshape(
                    self.episode_length * mini_batch_size * self.n_agents, -1)
                advantages_batch = advantages_batch.reshape(
                    self.episode_length * mini_batch_size * self.n_agents, -1)

                rnn_states_batch = rnn_states_batch.reshape(
                    mini_batch_size * self.n_agents, self.recurrent_N, -1)
                critic_rnn_states_batch = critic_rnn_states_batch.reshape(
                    mini_batch_size * self.n_agents, self.recurrent_N, -1)

                # obs_batch = obs_batch.reshape(
                #     self.episode_length * mini_batch_size * self.n_agents, -1)
                # parsed_obs_batch = [
                #     env_sentences[a_i]
                #     for env_sentences in parsed_obs_batch
                #     for a_i in range(self.n_agents)]

            else:
                policy_input_batch = policy_input_batch.reshape(
                    self.episode_length * mini_batch_size, self.n_agents, -1)
                critic_input_batch = critic_input_batch.reshape(
                    self.episode_length * mini_batch_size, self.n_agents, -1)
                env_actions_batch = env_actions_batch.reshape(
                    self.episode_length * mini_batch_size, self.n_agents, -1)
                comm_actions_batch = comm_actions_batch.reshape(
                    self.episode_length * mini_batch_size, self.n_agents, -1)
                env_action_log_probs_batch = env_action_log_probs_batch.reshape(
                    self.episode_length * mini_batch_size, self.n_agents, -1)
                comm_action_log_probs_batch = comm_action_log_probs_batch.reshape(
                    self.episode_length * mini_batch_size, self.n_agents, -1)
                value_preds_batch = value_preds_batch.reshape(
                    self.episode_length * mini_batch_size, self.n_agents, -1)
                returns_batch = returns_batch.reshape(
                    self.episode_length * mini_batch_size, self.n_agents, -1)
                masks_batch = masks_batch.reshape(
                    self.episode_length * mini_batch_size, self.n_agents, -1)
                advantages_batch = advantages_batch.reshape(
                    self.episode_length * mini_batch_size, self.n_agents, -1)

                # obs_batch = obs_batch.reshape(
                #     self.episode_length * mini_batch_size, self.n_agents, -1)

            yield policy_input_batch, critic_input_batch, rnn_states_batch, \
                critic_rnn_states_batch, env_actions_batch, comm_actions_batch, \
                env_action_log_probs_batch, comm_action_log_probs_batch, \
                value_preds_batch, returns_batch, masks_batch, advantages_batch
