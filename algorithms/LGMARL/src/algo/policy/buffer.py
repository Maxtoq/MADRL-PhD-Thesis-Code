import torch
import numpy as np


def _flatten(T, N, x):
    return x.reshape(T * N, *x.shape[2:])

def _cast(x):
    return x.transpose(1, 2, 0, 3).reshape(-1, *x.shape[3:])


class ACC_ReplayBuffer:

    def __init__(self, 
            args, n_agents, obs_dim, shared_obs_dim, env_act_dim, comm_act_dim):
        self.episode_length = args.episode_length
        self.n_parallel_envs = args.n_parallel_envs
        self.hidden_size = args.hidden_dim
        self.recurrent_N = args.policy_recurrent_N
        self.gamma = args.gamma
        self.gae_lambda = args.gae_lambda
        self.num_mini_batch = args.num_mini_batch
        self.data_chunk_length = args.data_chunk_length
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.shared_obs_dim = shared_obs_dim
        self.env_act_dim = env_act_dim
        self.comm_act_dim = comm_act_dim

        self.obs = np.zeros(
            (self.episode_length + 1, 
             self.n_parallel_envs, 
             n_agents, 
             self.obs_dim),
            dtype=np.float32)
        self.shared_obs = np.zeros(
            (self.episode_length + 1, 
             self.n_parallel_envs, 
             n_agents,
             self.shared_obs_dim),
            dtype=np.float32)

        self.rnn_states = np.zeros(
            (self.episode_length + 1, 
             self.n_parallel_envs, 
             n_agents,
             self.recurrent_N, 
             self.hidden_size), 
            dtype=np.float32)
        self.rnn_states_critic = np.zeros_like(self.rnn_states)

        self.value_preds = np.zeros(
            (self.episode_length + 1, self.n_parallel_envs, n_agents, 1),
            dtype=np.float32)
        self.returns = np.zeros(
            (self.episode_length + 1, self.n_parallel_envs, n_agents, 1),
            dtype=np.float32)

        self.env_actions = np.zeros(
            (self.episode_length, 
             self.n_parallel_envs, 
             n_agents, 
             self.env_act_dim),
            dtype=np.float32)
        self.env_action_log_probs = np.zeros(
            (self.episode_length, 
             self.n_parallel_envs, 
             n_agents, 
             self.env_act_dim),
            dtype=np.float32)

        self.comm_actions = np.zeros(
            (self.episode_length, 
             self.n_parallel_envs, 
             n_agents, 
             self.comm_act_dim),
            dtype=np.float32)
        self.comm_action_log_probs = np.zeros(
            (self.episode_length, 
             self.n_parallel_envs, 
             n_agents, 
             self.comm_act_dim),
            dtype=np.float32)

        self.rewards = np.zeros(
            (self.episode_length, self.n_parallel_envs, n_agents, 1), 
            dtype=np.float32)
        
        self.masks = np.ones(
            (self.episode_length + 1, self.n_parallel_envs, n_agents, 1), 
            dtype=np.float32)

        self.step = 0
    
    def reset_episode(self):
        self.obs = np.zeros((self.episode_length + 1, self.n_parallel_envs, self.n_agents, self.obs_dim), dtype=np.float32)
        self.shared_obs = np.zeros((self.episode_length + 1, self.n_parallel_envs, self.n_agents, self.shared_obs_dim), dtype=np.float32)
        self.rnn_states = np.zeros((self.episode_length + 1, self.n_parallel_envs, self.n_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        self.rnn_states_critic = np.zeros_like(self.rnn_states)
        self.value_preds = np.zeros((self.episode_length + 1, self.n_parallel_envs, self.n_agents, 1), dtype=np.float32)
        self.returns = np.zeros((self.episode_length + 1, self.n_parallel_envs, self.n_agents, 1), dtype=np.float32)
        self.env_actions = np.zeros((self.episode_length, self.n_parallel_envs, self.n_agents, self.env_act_dim), dtype=np.float32)
        self.env_action_log_probs = np.zeros((self.episode_length, self.n_parallel_envs, self.n_agents, self.env_act_dim), dtype=np.float32)
        self.comm_actions = np.zeros((self.episode_length, self.n_parallel_envs, self.n_agents, self.comm_act_dim), dtype=np.float32)
        self.comm_action_log_probs = np.zeros((self.episode_length, self.n_parallel_envs, self.n_agents, self.comm_act_dim), dtype=np.float32)
        self.rewards = np.zeros((self.episode_length, self.n_parallel_envs, self.n_agents, 1), dtype=np.float32)    
        self.masks = np.ones((self.episode_length + 1, self.n_parallel_envs, self.n_agents, 1), dtype=np.float32)
        self.step = 0

    def get_act_params(self):
        return self.obs[self.step], self.shared_obs[self.step], \
               self.rnn_states[self.step], self.rnn_states_critic[self.step], \
               self.masks[self.step]

    def insert_obs(self, obs, shared_obs):
        self.obs[self.step] = obs
        self.shared_obs[self.step] = shared_obs

    def insert_act(self, 
            rnn_states, rnn_states_critic, env_actions, env_action_log_probs, 
            comm_actions, comm_action_log_probs, value_preds, rewards, masks):
        self.rnn_states[self.step + 1] = rnn_states.copy()
        self.rnn_states_critic[self.step + 1] = rnn_states_critic.copy()
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
            delta = self.rewards[step] + self.gamma * value_normalizer.denormalize(
                self.value_preds[step + 1]) * self.masks[step + 1] \
                - value_normalizer.denormalize(self.value_preds[step])
            gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
            self.returns[step] = gae + value_normalizer.denormalize(
                self.value_preds[step])

    def recurrent_generator(self, advantages):
        """
        Yield training data for chunked RNN training.
        :param advantages: (np.ndarray) advantage estimates.
        """
        episode_length, n_rollout_threads, num_agents = self.rewards.shape[0:3]
        batch_size = n_rollout_threads * episode_length * num_agents
        data_chunks = batch_size // self.data_chunk_length  # [C=r*T*M/L]
        mini_batch_size = data_chunks // self.num_mini_batch

        rand = torch.randperm(data_chunks).numpy()
        sampler = [rand[i * mini_batch_size:(i + 1) * mini_batch_size] 
                    for i in range(self.num_mini_batch)]

        if len(self.shared_obs.shape) > 4:
            shared_obs = self.shared_obs[:-1].transpose(
                1, 2, 0, 3, 4, 5).reshape(-1, *self.shared_obs.shape[3:])
            obs = self.obs[:-1].transpose(
                1, 2, 0, 3, 4, 5).reshape(-1, *self.obs.shape[3:])
        else:
            shared_obs = _cast(self.shared_obs[:-1])
            obs = _cast(self.obs[:-1])

        env_actions = _cast(self.env_actions)
        env_action_log_probs = _cast(self.env_action_log_probs)
        comm_actions = _cast(self.comm_actions)
        comm_action_log_probs = _cast(self.comm_action_log_probs)
        advantages = _cast(advantages)
        value_preds = _cast(self.value_preds[:-1])
        returns = _cast(self.returns[:-1])
        masks = _cast(self.masks[:-1])
        rnn_states = self.rnn_states[:-1].transpose(
            1, 2, 0, 3, 4).reshape(-1, *self.rnn_states.shape[3:])
        rnn_states_critic = self.rnn_states_critic[:-1].transpose(
            1, 2, 0, 3, 4).reshape(-1, *self.rnn_states_critic.shape[3:])

        for indices in sampler:
            obs_batch = []
            shared_obs_batch = []
            rnn_states_batch = []
            rnn_states_critic_batch = []
            env_actions_batch = []
            comm_actions_batch = []
            value_preds_batch = []
            return_batch = []
            masks_batch = []
            old_env_action_log_probs_batch = []
            old_comm_action_log_probs_batch = []
            adv_targ = []

            for index in indices:

                ind = index * self.data_chunk_length
                # size [T+1 N M Dim]-->[T N M Dim]-->[N,M,T,Dim]-->[N*M*T,Dim]-->[L,Dim]
                obs_batch.append(obs[ind:ind + self.data_chunk_length])
                shared_obs_batch.append(
                    shared_obs[ind:ind + self.data_chunk_length])
                env_actions_batch.append(
                    env_actions[ind:ind + self.data_chunk_length])
                comm_actions_batch.append(
                    comm_actions[ind:ind + self.data_chunk_length])
                value_preds_batch.append(
                    value_preds[ind:ind + self.data_chunk_length])
                return_batch.append(returns[ind:ind + self.data_chunk_length])
                masks_batch.append(masks[ind:ind + self.data_chunk_length])
                old_env_action_log_probs_batch.append(
                    env_action_log_probs[ind:ind + self.data_chunk_length])
                old_comm_action_log_probs_batch.append(
                    comm_action_log_probs[ind:ind + self.data_chunk_length])
                adv_targ.append(advantages[ind:ind + self.data_chunk_length])
                # size [T+1 N M Dim]-->[T N M Dim]-->[N M T Dim]-->[N*M*T,Dim]-->[1,Dim]
                rnn_states_batch.append(rnn_states[ind])
                rnn_states_critic_batch.append(rnn_states_critic[ind])

            L, N = self.data_chunk_length, mini_batch_size

            # These are all from_numpys of size (L, N, Dim)
            obs_batch = np.stack(obs_batch, axis=1)
            shared_obs_batch = np.stack(shared_obs_batch, axis=1)
            env_actions_batch = np.stack(env_actions_batch, axis=1)
            comm_actions_batch = np.stack(comm_actions_batch, axis=1)
            value_preds_batch = np.stack(value_preds_batch, axis=1)
            return_batch = np.stack(return_batch, axis=1)
            masks_batch = np.stack(masks_batch, axis=1)
            old_env_action_log_probs_batch = np.stack(
                old_env_action_log_probs_batch, axis=1)
            old_comm_action_log_probs_batch = np.stack(
                old_comm_action_log_probs_batch, axis=1)
            adv_targ = np.stack(adv_targ, axis=1)

            # States is just a (N, -1) from_numpy
            rnn_states_batch = np.stack(rnn_states_batch).reshape(
                N, *self.rnn_states.shape[3:])
            rnn_states_critic_batch = np.stack(rnn_states_critic_batch).reshape(
                    N, *self.rnn_states_critic.shape[3:])

            # Flatten the (L, N, ...) from_numpys to (L * N, ...)
            obs_batch = _flatten(L, N, obs_batch)
            shared_obs_batch = _flatten(L, N, shared_obs_batch)
            env_actions_batch = _flatten(L, N, env_actions_batch)
            comm_actions_batch = _flatten(L, N, comm_actions_batch)
            value_preds_batch = _flatten(L, N, value_preds_batch)
            return_batch = _flatten(L, N, return_batch)
            masks_batch = _flatten(L, N, masks_batch)
            old_env_action_log_probs_batch = _flatten(
                L, N, old_env_action_log_probs_batch)
            old_comm_action_log_probs_batch = _flatten(
                L, N, old_comm_action_log_probs_batch)
            adv_targ = _flatten(L, N, adv_targ)

            yield obs_batch, shared_obs_batch, rnn_states_batch, rnn_states_critic_batch, \
                  env_actions_batch, comm_actions_batch, value_preds_batch, \
                  return_batch, masks_batch, old_env_action_log_probs_batch,\
                  old_comm_action_log_probs_batch, adv_targ