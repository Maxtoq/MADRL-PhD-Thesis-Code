import random


class PerfectComm:

    def __init__(self, lang_learner, prob_send_message=1.0):
        self.lang_learner = lang_learner
        self.prob_send_message = prob_send_message

    def _rand_filter_messages(self, messages):
        """
        Randomly filter out perfect messages.
        :param messages (list(list(list(str)))): Perfect messages, ordered by
            environment, by agent.

        :return filtered_broadcast (list(list(str))): Filtered message to 
            broadcast, one for each environment.
        """
        filtered_broadcast = []
        for env_messages in messages:
            env_broadcast = []
            for message in env_messages:
                if random.random() < self.prob_send_message:
                    env_broadcast.extend(message)
            filtered_broadcast.append(env_broadcast)
        return filtered_broadcast

    def comm_step(self, obs, lang_contexts, perfect_messages):
        """
        Perform a communication step.

        :param obs (np.ndarray): Observations, dim=(n_parallel_envs, n_agents, 
            obs_dim).
        :param lang_contexts (np.ndarray): Language contexts from last step,
            dim=(n_parallel_envs, context_dim).
        :param perfect messages (list(list(list(str)))): Perfect messages,
            ordered by environment, by agent.

        :return broadcasts (list(list(str))): Broadcasted messages for each
            environment.
        :return next_contents (np.ndarray): Language contexts for next step,
            dim=(n_envs, context_dim).
        """
        # Determines the content of the broadcasted message
        broadcasts = self._rand_filter_messages(perfect_messages)
        
        # Compute next context
        next_contexts = self.lang_learner.encode_sentences(broadcasts)

        return broadcasts, perfect_messages, next_contexts.detach().cpu().numpy()

    def store_rewards(self, message_rewards, token_rewards):
        """
        Send rewards for each sentences to the buffer to compute returns.

        :param message_rewards (np.ndarray): Rewards for each generated 
            sentence, dim=(batch_size, )
        :param klpretrain_rewards (np.ndarray): Penalties for diverging from 
            pre-trained decoder, dim=(seq_len, batch_size, 1)
        """
        pass

    def reset_buffer(self):
        pass

    def train(self, warmup=False):
        return {}

    def get_save_dict(self):
        return {}

    def prep_rollout(self, device=None):
        pass

    def prep_training(self):
        pass