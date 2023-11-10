import os
import csv
import json


class CommunicationLogger:

    def __init__(self, save_dir):
        self.observations = []
        self.generated_messages = []
        self.perfect_messages = []
        self.broadcasts = []
        # self.kl_pens = []
        self.env_rewards = []

        # Create csv file
        self.csv_path = os.path.join(save_dir, "comm_logs.csv")
        with open(self.csv_path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow([
                "Generated Message", 
                "Perfect Message", 
                "Broadcasted Message",
                # "KL Penalty",
                "Env Reward",
                "Observation"])

    def store_messages(self, 
            obs, gen_mess, perf_mess, broadcasts, kl_pen=None):
        """
        Store messages and corresponding observations.

        :param obs (np.ndarray): Observations for each agent in each parallel 
            environment, dim=(n_parallel_envs, n_agents, obs_dim).
        :param gen_mess (list(list(list(str)))): Generated messages, ordered by
            parallel environment.
        :param perf_mess (list(list(str))): "Perfect" messages, ordered by
            parallel environment.
        :param broadcasts (list(list(list(str)))): Broadcasted messages, one per
            parallel environment.
        :param kl_pen (np.ndarray): Sum of KL penalties for each messages,
            dim=(n_parallel_envs * n_agents, )
        """
        n_parallel_envs = obs.shape[0]
        n_agents = obs.shape[1]
        self.observations += obs.reshape(
            n_parallel_envs * n_agents, -1).tolist()
        for e_i in range(n_parallel_envs):
            self.generated_messages += gen_mess[e_i]
            self.perfect_messages += perf_mess[e_i]
            self.broadcasts += [broadcasts[e_i]] * n_agents
        # if kl_pen is not None:
        #     self.kl_pens += kl_pen.tolist()

    def store_rewards(self, rewards):
        """
        Store message rewards.

        :param rewards (np.ndarray): Rewards for each message in each parallel
            environment, dim=(n_parallel_envs, n_agents).
        """
        n_parallel_envs = rewards.shape[0]
        n_agents = rewards.shape[1]
        self.env_rewards += rewards.flatten().tolist()

    def save(self):
        with open(self.csv_path, 'a+', newline='') as f:
            w = csv.writer(f)
            for o, gm, pm, br, er in zip(
                    self.observations, 
                    self.generated_messages,
                    self.perfect_messages,
                    self.broadcasts,
                    # self.kl_pens,
                    self.env_rewards):
                w.writerow([
                    " ".join(gm),
                    " ".join(pm),
                    " ".join(br),
                    # str(kl),
                    str(er),
                    " ".join(str(o_i) for o_i in o)])
        
        self.observations = []
        self.generated_messages = []
        self.perfect_messages = []
        self.broadcasts = []
        self.kl_pens = []
        self.env_rewards = []
