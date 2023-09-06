import numpy as np
import pandas as pd

from tensorboardX import SummaryWriter


class Logger():
    """
    Class for our logger that saves training and evaluation data during 
    training.
    :param args: (argparse.Namespace) training arguments
    :param log_dir_path: (str) path of directory where logs are saved
    """
    def __init__(self, args, log_dir_path):
        self.log_dir_path = log_dir_path
        self.ep_length = args.episode_length
        self.log_tensorboard = args.log_tensorboard
        self.n_parrallel_envs = args.n_rollout_threads
        self.do_eval = args.do_eval

        self.train_data = {
            "Step": [],
            "Episode return": [],
            "Success": [],
            "Episode length": []
        }
        self.returns = np.zeros(self.n_parrallel_envs)
        self.success = [False] * self.n_parrallel_envs
        self.ep_lengths = np.ones(self.n_parrallel_envs) * self.ep_length

        self.eval_data = {
            "Step": [],
            "Mean return": [],
            "Success rate": [],
            "Mean episode length": []
        }

        if self.log_tensorboard:
            self.log_tb = SummaryWriter(str(self.log_dir_path))

    def reset_episode(self):
        self.returns = np.zeros(self.n_parrallel_envs)
        self.success = [False] * self.n_parrallel_envs
        self.ep_lengths = np.ones(self.n_parrallel_envs) * self.ep_length

    def count_returns(self, step, rewards, dones):
        global_rewards = rewards.mean(axis=1)
        global_dones = dones.all(axis=1)
        for e_i in range(self.n_parrallel_envs):
            if not self.success[e_i]:
                self.returns[e_i] += global_rewards[e_i]
                self.extr_returns[e_i] += global_extr_rewards[e_i]
                self.intr_returns[e_i] += global_intr_rewards[e_i]
                if global_dones[e_i]:
                    self.success[e_i] = True
                    self.ep_lengths[e_i] = step + 1

    def log_train(self, step, train_losses):
        """
        Log training data.
        :param step: (int) Step number.
        :param train_losses: (dict/tuple) Dictionaries containing training losses.

        :output n_don_steps: (int) Number of steps performed in all parallel 
            environments.
        """
        n_done_steps = 0
        for e_i in range(self.n_parrallel_envs):
            n_done_steps += self.ep_lengths[e_i]
            # Log dict
            self.train_data["Step"].append(n_done_steps + step)
            self.train_data["Episode return"].append(self.returns[e_i])
            self.train_data["Episode extrinsic return"].append(
                self.extr_returns[e_i])
            self.train_data["Episode intrinsic return"].append(
                self.intr_returns[e_i])
            self.train_data["Success"].append(int(self.success[e_i]))
            self.train_data["Episode length"].append(self.ep_lengths[e_i])
            # Log Tensorboard
            if self.log_tensorboard:
                self.log_tb.add_scalar(
                    'agent0/episode_return', 
                    self.train_data["Episode return"][-1], 
                    self.train_data["Step"][-1])
                self.log_tb.add_scalar(
                    'agent0/episode_ext_return', 
                    self.train_data["Episode extrinsic return"][-1], 
                    self.train_data["Step"][-1])
                self.log_tb.add_scalar(
                    'agent0/episode_int_return', 
                    self.train_data["Episode intrinsic return"][-1], 
                    self.train_data["Step"][-1])

        # Log losses
        if self.log_tensorboard:
            if type(train_losses) is tuple:
                losses = {
                    "value_loss": np.mean(
                        [t["value_loss"] for t in train_losses[0]]),
                    "policy_loss": np.mean(
                        [t["policy_loss"] for t in train_losses[0]]),
                    "rnd_loss": train_losses[1]["rnd_loss"],
                    "e3b_loss": train_losses[1]["e3b_loss"]}
            else:
                losses = {
                    "value_loss": np.mean(
                        [t["value_loss"] for t in train_losses]),
                    "policy_loss": np.mean(
                        [t["policy_loss"] for t in train_losses])}
            self.log_tb.add_scalars(
                'agent0/losses', losses, step + n_done_steps)

        return int(n_done_steps)

    def log_eval(self, step, mean_return, success_rate, mean_ep_len):
        self.eval_data["Step"].append(step)
        self.eval_data["Mean return"].append(mean_return)
        self.eval_data["Success rate"].append(success_rate)
        self.eval_data["Mean episode length"].append(mean_ep_len)

    def save(self):
        train_df = pd.DataFrame(self.train_data)
        train_df.to_csv(str(self.log_dir_path / 'training_data.csv'))
        if self.do_eval:
            eval_df = pd.DataFrame(self.eval_data)
            eval_df.to_csv(str(self.log_dir_path / 'evaluation_data.csv'))

    def save_n_close(self):
        self.save()
        if self.log_tensorboard:
            self.log_tb.export_scalars_to_json(
                str(self.log_dir_path / 'summary.json'))
            self.log_tb.close()