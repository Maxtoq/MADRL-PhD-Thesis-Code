

class CommunicationLogger:

    def __init__(self, save_dir):
        self.save_dir = save_dir

        self.observations = []
        self.generated_messages = []
        self.perfect_messages = []
        self.message_kl_pens = []
        self.message_rewards = []

    def store_messages(self, 
            obs, gen_mess=None, perf_mess=None, kl_pen=None):
        """
        Store messages and corresponding observations.

        :param obs (np.ndarray): Observations for each agent in each parallel 
            environment, dim=(n_parallel_envs, n_agents, obs_dim).
        :param gen_mess (list(list(list(str)))): Generated messages, ordered by
            parallel environment, default None.
        :param perf_mess (list(list(list(str)))): "Perfect" messages, ordered by
            parallel environment, default None.
        :param kl_pen (np.ndarray): Sum of KL penalties for each messages,
            dim=(n_parallel_envs * n_agents, )
        """
        self.observations.append(obs)
        if gen_mess is not None:
            self.generated_messages.append(gen_mess)
        if perf_mess is not None:
            self.perfect_messages.append(perf_mess)
        if kl_pen is not None:
            n_parallel_envs = obs.shape[0]
            n_agents = obs.shape[1]
            self.message_kl_pens.append(
                kl_pen.reshape(n_parallel_envs, n_agents))

    def store_rewards(self, rewards):
        """
        Store message rewards.

        :param rewards (np.ndarray): Rewards for each message in each parallel
            environment, dim=(n_parallel_envs, n_agents).
        """
        self.message_rewards.append(rewards)

    def save(self):
        # TODO checker qu'on a bien log tout, et save dans un fichier
        pass
