import torch
import numpy as np

from src.log.comm_logs import CommunicationLogger


def _flatten(T, N, x):
    return x.reshape(T * N, *x.shape[2:])

def _cast(x):
    return x.transpose(1, 2, 0, 3).reshape(-1, *x.shape[3:])




class MessageSampler():

    def __init__(self):
        self.observed_messages = {}
        self.n_messages = 0

    def add_messages(self, messages):
        messages = messages.reshape(-1, messages.shape[-1])
        for mess in messages:
            # key = mess.tobytes()
            key = str(mess)
            if key in self.observed_messages:
                self.observed_messages[key] += 1
            else:
                self.observed_messages[key] = 1
            self.n_messages += 1

    def get_message_probs(self, message_batch):
        batch_size = message_batch.shape[0]

        # Get number of occurence of each message
        n_occs = np.zeros(batch_size)
        for m_i, mess in enumerate(message_batch):
            # key = mess.tobytes()
            key = str(mess)
            if key in self.observed_messages:
                n_occs[m_i] = self.observed_messages[key]
            else:
                n_occs[m_i] = 1 / self.n_messages
        
        # Compute probalities
        probs = 1 / n_occs
        probs = np.exp(probs) / np.exp(probs).sum()

        return probs


class ReplayBuffer:

    def __init__(self, 
            args, n_agents, obs_dim, joint_obs_dim, env_act_dim, 
            comm_act_dim, max_message_len, log_dir=None):
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.joint_obs_dim = joint_obs_dim
        self.env_act_dim = env_act_dim
        self.comm_act_dim = comm_act_dim
        self.max_message_len = max_message_len

        self.rollout_length = args.rollout_length
        self.n_parallel_envs = args.n_parallel_envs
        self.hidden_size = args.hidden_dim
        self.recurrent_N = args.policy_recurrent_N
        self.gamma = args.gamma
        self.gae_lambda = args.gae_lambda
        self.n_mini_batch = args.n_mini_batch
        self.data_chunk_length = args.data_chunk_length
        self.share_params = args.share_params

        # self.lang_imp_sample = args.lang_imp_sample
        # self.message_sampler = MessageSampler()

        self.obs = np.zeros(
            (self.rollout_length + 1, 
             self.n_parallel_envs, 
             self.n_agents, 
             self.obs_dim),
            dtype=np.float32)
        self.joint_obs = np.zeros(
            (self.rollout_length + 1, 
             self.n_parallel_envs, 
             self.n_agents,
             self.joint_obs_dim),
            dtype=np.float32)

        self.obs_enc_rnn_states = np.zeros(
            (self.rollout_length + 1, 
             self.n_parallel_envs, 
             self.n_agents,
             self.recurrent_N, 
             self.hidden_size), 
            dtype=np.float32)
        self.joint_obs_enc_rnn_states = np.zeros_like(self.obs_enc_rnn_states)
        self.comm_enc_rnn_states = np.zeros_like(self.obs_enc_rnn_states)

        self.act_value_preds = np.zeros(
            (self.rollout_length + 1, self.n_parallel_envs, self.n_agents, 1),
            dtype=np.float32)
        self.act_returns = np.zeros(
            (self.rollout_length + 1, self.n_parallel_envs, self.n_agents, 1),
            dtype=np.float32)

        self.comm_value_preds = np.zeros(
            (self.rollout_length + 1, self.n_parallel_envs, self.n_agents, 1),
            dtype=np.float32)
        self.comm_returns = np.zeros(
            (self.rollout_length + 1, self.n_parallel_envs, self.n_agents, 1),
            dtype=np.float32)

        self.env_actions = np.zeros(
            (self.rollout_length, 
             self.n_parallel_envs, 
             self.n_agents, 
             self.env_act_dim),
            dtype=np.float32)
        self.env_action_log_probs = np.zeros(
            (self.rollout_length, 
             self.n_parallel_envs, 
             self.n_agents, 
             self.env_act_dim),
            dtype=np.float32)

        self.comm_actions = np.zeros(
            (self.rollout_length, 
             self.n_parallel_envs, 
             self.n_agents, 
             self.comm_act_dim),
            dtype=np.float32)
        self.comm_action_log_probs = np.zeros(
            (self.rollout_length, 
             self.n_parallel_envs, 
             self.n_agents, 
             1),
            dtype=np.float32)

        self.act_rewards = np.zeros(
            (self.rollout_length, self.n_parallel_envs, self.n_agents, 1), 
            dtype=np.float32)

        # Communication
        self.comm_rewards = np.zeros(
            (self.rollout_length, self.n_parallel_envs, self.n_agents, 1), 
            dtype=np.float32)
        self.gen_comm = np.zeros(
            (self.rollout_length, self.n_parallel_envs, self.n_agents, 1), 
            dtype=np.float32)
        
        self.masks = np.ones(
            (self.rollout_length + 1, self.n_parallel_envs, self.n_agents, 1), 
            dtype=np.float32)

        # Language data
        self.perf_messages = np.zeros(
            (self.rollout_length + 1, 
             self.n_parallel_envs, 
             self.n_agents, 
             self.max_message_len),
            dtype=np.int32)
        self.perf_broadcasts = []

        self.step = 0

        if args.log_comm:
            print("LOGGING COMUNICATION IN", log_dir)
            self.comm_logger = CommunicationLogger(log_dir)
        else:
            self.comm_logger = None
    
    def reset(self):
        self.obs = np.zeros((self.rollout_length + 1, self.n_parallel_envs, self.n_agents, self.obs_dim), dtype=np.float32)
        self.joint_obs = np.zeros((self.rollout_length + 1, self.n_parallel_envs, self.n_agents, self.joint_obs_dim), dtype=np.float32)
        self.obs_enc_rnn_states = np.zeros((self.rollout_length + 1, self.n_parallel_envs, self.n_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        self.joint_obs_enc_rnn_states = np.zeros_like(self.obs_enc_rnn_states)
        self.comm_enc_rnn_states = np.zeros_like(self.obs_enc_rnn_states)
        self.act_value_preds = np.zeros((self.rollout_length + 1, self.n_parallel_envs, self.n_agents, 1), dtype=np.float32)
        self.act_returns = np.zeros((self.rollout_length + 1, self.n_parallel_envs, self.n_agents, 1), dtype=np.float32)
        self.comm_value_preds = np.zeros((self.rollout_length + 1, self.n_parallel_envs, self.n_agents, 1), dtype=np.float32)
        self.comm_returns = np.zeros((self.rollout_length + 1, self.n_parallel_envs, self.n_agents, 1), dtype=np.float32)
        self.env_actions = np.zeros((self.rollout_length, self.n_parallel_envs, self.n_agents, self.env_act_dim), dtype=np.float32)
        self.env_action_log_probs = np.zeros((self.rollout_length, self.n_parallel_envs, self.n_agents, self.env_act_dim), dtype=np.float32)
        self.comm_actions = np.zeros((self.rollout_length, self.n_parallel_envs, self.n_agents, self.comm_act_dim), dtype=np.float32)
        self.comm_action_log_probs = np.zeros((self.rollout_length, self.n_parallel_envs, self.n_agents, 1), dtype=np.float32)
        self.act_rewards = np.zeros((self.rollout_length, self.n_parallel_envs, self.n_agents, 1), dtype=np.float32) 
        self.comm_rewards = np.zeros((self.rollout_length, self.n_parallel_envs, self.n_agents, 1), dtype=np.float32)
        self.gen_comm = np.zeros((self.rollout_length, self.n_parallel_envs, self.n_agents, 1), dtype=np.float32)
        self.masks = np.ones((self.rollout_length + 1, self.n_parallel_envs, self.n_agents, 1), dtype=np.float32)
        self.perf_messages = np.zeros((self.rollout_length + 1, self.n_parallel_envs, self.n_agents, self.max_message_len), dtype=np.int32)
        self.perf_broadcasts = []
        self.step = 0

    def start_new_episode(self):
        if self.comm_logger is not None:
            self.comm_logger.log(
                self.obs, 
                self.comm_rewards, 
                self.comm_returns, 
                self.perf_messages, 
                self.perf_broadcasts)
        self.obs[0] = self.obs[-1].copy()
        self.joint_obs[0] = self.joint_obs[-1].copy()
        self.obs_enc_rnn_states[0] = self.obs_enc_rnn_states[-1].copy()
        self.joint_obs_enc_rnn_states[0] = self.joint_obs_enc_rnn_states[-1].copy()
        self.comm_enc_rnn_states[0] = self.comm_enc_rnn_states[-1].copy()
        self.masks[0] = self.masks[-1].copy()
        self.perf_messages[0] = self.perf_messages[-1].copy()
        self.perf_broadcasts = [self.perf_broadcasts[-1]]
        self.step = 0

    def insert_obs(self, obs, joint_obs, perf_messages, perf_broadcasts):
        self.obs[self.step] = obs
        self.joint_obs[self.step] = joint_obs
        self.perf_messages[self.step] = perf_messages
        self.perf_broadcasts.append(perf_broadcasts)
        # if self.lang_imp_sample:
        #     self.message_sampler.add_messages(perf_messages)

    def get_act_params(self):
        return self.obs[self.step], self.joint_obs[self.step], \
               self.obs_enc_rnn_states[self.step], \
               self.joint_obs_enc_rnn_states[self.step], \
               self.comm_enc_rnn_states[self.step], \
               self.masks[self.step], self.perf_messages[self.step], \
               self.perf_broadcasts[self.step]

    def insert_act(self, 
            obs_enc_rnn_states, joint_obs_enc_rnn_states, comm_enc_rnn_states, 
            env_actions, env_action_log_probs, comm_actions, comm_action_log_probs, 
            act_value_preds, comm_value_preds, act_rewards, masks, comm_rewards, 
            gen_comm):
        self.obs_enc_rnn_states[self.step + 1] = obs_enc_rnn_states.copy()
        self.joint_obs_enc_rnn_states[self.step + 1] = joint_obs_enc_rnn_states.copy()
        self.comm_enc_rnn_states[self.step + 1] = comm_enc_rnn_states.copy()
        self.env_actions[self.step] = env_actions.copy()
        self.env_action_log_probs[self.step] = env_action_log_probs.copy()
        self.comm_actions[self.step] = comm_actions.copy()
        self.comm_action_log_probs[self.step] = comm_action_log_probs.copy()
        self.act_value_preds[self.step] = act_value_preds.copy()
        self.comm_value_preds[self.step] = comm_value_preds.copy()
        self.act_rewards[self.step] = act_rewards.copy()
        self.masks[self.step + 1] = masks.copy()
        self.comm_rewards[self.step] = comm_rewards.copy()
        if gen_comm is not None:
            self.gen_comm[self.step] = gen_comm.copy()
        self.step += 1

    def compute_returns(self,
            next_act_value, next_comm_value, act_value_normalizer, 
            comm_value_normalizer):
        """
        Compute returns either as discounted sum of rewards, or using GAE.
        :param next_act_value: (np.ndarray) action value predictions for the 
            step after the last episode step.
        :param next_comm_value: (np.ndarray) comm value predictions for the 
            step after the last episode step.
        :param act_value_normalizer: (ValueNorm) Value normalizer instance.
        :param comm_value_normalizer: (ValueNorm) Value normalizer instance.
        """
        # Messages get environmnent reward
        self.comm_rewards += self.act_rewards

        self.act_value_preds[-1] = next_act_value
        self.comm_value_preds[-1] = next_comm_value

        act_gae = 0
        comm_gae = 0
        for step in reversed(range(self.act_rewards.shape[0])):
            delta = self.act_rewards[step] + self.gamma \
                * act_value_normalizer.denormalize(
                    self.act_value_preds[step + 1]) * self.masks[step + 1] \
                    - act_value_normalizer.denormalize(self.act_value_preds[step])
            act_gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] \
                * act_gae
            self.act_returns[step] = act_gae + act_value_normalizer.denormalize(
                self.act_value_preds[step])

            delta = self.comm_rewards[step] + self.gamma \
                * comm_value_normalizer.denormalize(
                    self.comm_value_preds[step + 1]) * self.masks[step + 1] \
                    - comm_value_normalizer.denormalize(self.comm_value_preds[step])
            comm_gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] \
                * comm_gae
            self.comm_returns[step] = comm_gae + comm_value_normalizer.denormalize(
                self.comm_value_preds[step])

    # def _get_mess_sampl_probs(self, messages):
    #     if len(messages.shape) == 3:
    #         probs = np.zeros(messages.shape[:-1])
    #         for a_i in range(self.n_agents):
    #             probs[:, a_i] = self.message_sampler.get_message_probs(messages[:, a_i])
    #         probs = probs.reshape(messages.shape[0], self.n_agents)
    #     else:
    #         probs = self.message_sampler.get_message_probs(messages)
    #     return probs

    def recurrent_policy_generator(self, act_advt, comm_advt):
        """
        Generates sample for policy training.
        :param act_advt: (np.ndarray) Env actions advantages.
        :param comm_advt: (np.ndarray) Communication actions advantages.
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
            obs_batch = self.obs[:-1, ids] # T x mini_batch_size x N_a x obs_dim
            joint_obs_batch = self.joint_obs[:-1, ids]
            obs_enc_rnn_states_batch = self.obs_enc_rnn_states[0, ids]
            joint_obs_enc_rnn_states_batch = self.joint_obs_enc_rnn_states[0, ids]
            comm_enc_rnn_states_batch = self.comm_enc_rnn_states[0, ids]
            env_actions_batch = self.env_actions[:, ids]
            comm_actions_batch = self.comm_actions[:, ids]
            env_action_log_probs_batch = self.env_action_log_probs[:, ids]
            comm_action_log_probs_batch = self.comm_action_log_probs[:, ids]
            act_value_preds_batch = self.act_value_preds[:-1, ids]
            act_returns_batch = self.act_returns[:-1, ids]
            comm_value_preds_batch = self.comm_value_preds[:-1, ids]
            comm_returns_batch = self.comm_returns[:-1, ids]
            masks_batch = self.masks[:-1, ids]

            gen_comm_batch = self.gen_comm[:, ids]

            act_advt_batch = act_advt[:, ids]
            comm_advt_batch = comm_advt[:, ids]

            perf_messages_batch = self.perf_messages[:-1, ids]
            # Flatten the broadcasts first dimension (env_step) and get only 
            # broadcasts from envs[ids]
            perf_broadcasts_batch = [
                step_sentences[i] 
                for step_sentences in self.perf_broadcasts[:-1]
                for i in ids]
            
            # Same for 
            # perf_messages_batch = [
            #     step_sentences[i] 
            #     for step_sentences in self.perf_messages[:-1]
            #     for i in ids]

            # if self.share_params:
            #     obs_batch = obs_batch.reshape(
            #         self.rollout_length * mini_batch_size * self.n_agents, -1)
            #     joint_obs_batch = joint_obs_batch.reshape(
            #         self.rollout_length * mini_batch_size * self.n_agents, -1)
            #     env_actions_batch = env_actions_batch.reshape(
            #         self.rollout_length * mini_batch_size * self.n_agents, -1)
            #     comm_actions_batch = comm_actions_batch.reshape(
            #         self.rollout_length * mini_batch_size * self.n_agents, -1)
            #     env_action_log_probs_batch = env_action_log_probs_batch.reshape(
            #         self.rollout_length * mini_batch_size * self.n_agents, -1)
            #     comm_action_log_probs_batch = comm_action_log_probs_batch.reshape(
            #         self.rollout_length * mini_batch_size * self.n_agents, -1)
            #     act_value_preds_batch = act_value_preds_batch.reshape(
            #         self.rollout_length * mini_batch_size * self.n_agents, -1)
            #     act_returns_batch = act_returns_batch.reshape(
            #         self.rollout_length * mini_batch_size * self.n_agents, -1)
            #     comm_value_preds_batch = comm_value_preds_batch.reshape(
            #         self.rollout_length * mini_batch_size * self.n_agents, -1)
            #     comm_returns_batch = comm_returns_batch.reshape(
            #         self.rollout_length * mini_batch_size * self.n_agents, -1)
            #     masks_batch = masks_batch.reshape(
            #         self.rollout_length * mini_batch_size * self.n_agents, -1)
            #     gen_comm_batch = gen_comm_batch.reshape(
            #         self.rollout_length * mini_batch_size * self.n_agents, -1)
            #     act_advt_batch = act_advt_batch.reshape(
            #         self.rollout_length * mini_batch_size * self.n_agents, -1)
            #     comm_advt_batch = comm_advt_batch.reshape(
            #         self.rollout_length * mini_batch_size * self.n_agents, -1)

            #     obs_enc_rnn_states_batch = obs_enc_rnn_states_batch.reshape(
            #         mini_batch_size * self.n_agents, self.recurrent_N, -1)
            #     joint_obs_enc_rnn_states_batch = joint_obs_enc_rnn_states_batch.reshape(
            #         mini_batch_size * self.n_agents, self.recurrent_N, -1)
            #     comm_enc_rnn_states_batch = comm_enc_rnn_states_batch.reshape(
            #         mini_batch_size * self.n_agents, self.recurrent_N, -1)

            #     perf_messages_batch = perf_messages_batch.reshape(
            #         self.rollout_length * mini_batch_size * self.n_agents, -1)
            #     # Flatten all perf_broadcasts
            #     perf_broadcasts_batch = [
            #         env_sentences[a_i]
            #         for env_sentences in perf_broadcasts_batch
            #         for a_i in range(self.n_agents)]
            #     # perf_messages_batch = [
            #     #     env_sentences[a_i]
            #     #     for env_sentences in perf_messages_batch
            #     #     for a_i in range(self.n_agents)]

            # else:
            obs_batch = obs_batch.reshape(
                self.rollout_length * mini_batch_size, self.n_agents, -1)
            joint_obs_batch = joint_obs_batch.reshape(
                self.rollout_length * mini_batch_size, self.n_agents, -1)
            env_actions_batch = env_actions_batch.reshape(
                self.rollout_length * mini_batch_size, self.n_agents, -1)
            comm_actions_batch = comm_actions_batch.reshape(
                self.rollout_length * mini_batch_size, self.n_agents, -1)
            env_action_log_probs_batch = env_action_log_probs_batch.reshape(
                self.rollout_length * mini_batch_size, self.n_agents, -1)
            comm_action_log_probs_batch = comm_action_log_probs_batch.reshape(
                self.rollout_length * mini_batch_size, self.n_agents, -1)
            act_value_preds_batch = act_value_preds_batch.reshape(
                self.rollout_length * mini_batch_size, self.n_agents, -1)
            act_returns_batch = act_returns_batch.reshape(
                self.rollout_length * mini_batch_size, self.n_agents, -1)
            comm_value_preds_batch = comm_value_preds_batch.reshape(
                self.rollout_length * mini_batch_size, self.n_agents, -1)
            comm_returns_batch = comm_returns_batch.reshape(
                self.rollout_length * mini_batch_size, self.n_agents, -1)
            masks_batch = masks_batch.reshape(
                self.rollout_length * mini_batch_size, self.n_agents, -1)
            gen_comm_batch = gen_comm_batch.reshape(
                self.rollout_length * mini_batch_size, self.n_agents, -1)
            act_advt_batch = act_advt_batch.reshape(
                self.rollout_length * mini_batch_size, self.n_agents, -1)
            comm_advt_batch = comm_advt_batch.reshape(
                self.rollout_length * mini_batch_size, self.n_agents, -1)

            perf_messages_batch = perf_messages_batch.reshape(
                self.rollout_length * mini_batch_size, self.n_agents, -1)

        # Get probabilities of sampling each message for language training
        # if self.lang_imp_sample:
        #     mess_sampling_probs = self._get_mess_sampl_probs(
        #         perf_messages_batch)
        # else:
        # mess_sampling_probs = np.zeros_like(perf_messages_batch)

        yield obs_batch, joint_obs_batch, obs_enc_rnn_states_batch, \
            joint_obs_enc_rnn_states_batch, comm_enc_rnn_states_batch, \
            env_actions_batch, comm_actions_batch, \
            env_action_log_probs_batch, comm_action_log_probs_batch, \
            act_value_preds_batch, comm_value_preds_batch, act_returns_batch, \
            comm_returns_batch, masks_batch, act_advt_batch, comm_advt_batch, \
            gen_comm_batch, perf_messages_batch, perf_broadcasts_batch
