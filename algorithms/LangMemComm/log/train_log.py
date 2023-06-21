import numpy as np

from tensorboardX import SummaryWriter


class Logger():
    """
    Class for our logger that saves training and evaluation data during 
    training.
    :param log_dir_path: (str) path of directory where logs are saved
    :param do_tensorboard: (bool) whether to log in tensorboard
    """
    def __init__(self, log_dir_path, ep_length, n_parrallel_envs=1, do_tensorboard=True):
        self.log_dir_path = log_dir_path
        self.do_tensorboard = do_tensorboard
        self.n_parrallel_envs = n_parrallel_envs

        self.train_data = {
            "Step": [],
            "Episode return": [],
            "Success": [],
            "Episode length": []
        }
        self.returns = np.zeros(self.n_parrallel_envs)
        self.success = [False] * self.n_parrallel_envs
        self.ep_lengths = np.ones(self.n_parrallel_envs) * ep_length

        self.eval_data = {
            "Step": [],
            "Mean return": [],
            "Success rate": [],
            "Mean episode length": []
        }

        if self.do_tensorboard:
            self.log_tb = SummaryWriter(str(self.log_dir_path))

    def reset_episode(self):
        self.returns = np.zeros(self.n_parrallel_envs)
        self.success = [False] * self.n_parrallel_envs
        self.ep_lengths = np.ones(self.n_parrallel_envs) * ep_length

    def count_returns(self, step, rewards, dones):
        global_rewards = rewards.mean(axis=1)
        global_dones = dones.all(axis=1)
        for e_i in range(self.n_parrallel_envs):
            if not self.success[e_i]:
                self.returns[e_i] += global_rewards[e_i, 0]
                if global_dones[e_i]:
                    self.success[e_i] = True
                    self.ep_lengths[e_i] = step + 1

    def log(self, step):
        """
        Log training data.
        :param rewards: (numpy.ndarray) List of rewards in multiple parrallel
            environments
        :param dones: (numpy.ndarray) List of dones in multiple parrallel
            environments

        :output tot_steps: (int) number of steps performed in all parallel 
            environments
        """
        n_done_steps = 0
        for e_i in range(self.n_parrallel_envs):
            n_done_steps += self.ep_lengths[e_i]
            # Log dict
            self.train_data["Step"].append(n_done_steps + step)
            self.train_data["Episode return"].append(self.returns[e_i])
            self.train_data["Success"].append(int(self.success[e_i]))
            self.train_data["Episode length"].append(self.ep_lengths[e_i])
            # Log Tensorboard
            self.log_tb.add_scalar(
                'agent0/episode_return', 
                self.train_data["Episode return"][-1], 
                self.train_data["Step"][-1])
        return int(n_done_steps)