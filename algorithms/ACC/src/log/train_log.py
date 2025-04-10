import os
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
    def __init__(self, args, log_dir_path, n_steps_done=0):
        self.log_dir_path = log_dir_path
        self.max_ep_length = args.episode_length
        self.log_tensorboard = args.log_tensorboard
        self.n_parrallel_envs = args.n_parallel_envs
        self.do_eval = args.do_eval

        self.train_data = {
            "Step": [],
            "Episode return": [],
            "Success": [],
            "Episode length": []
        }
        self.returns = np.zeros(self.n_parrallel_envs)
        self.success = [False] * self.n_parrallel_envs
        self.ep_lengths = np.zeros(self.n_parrallel_envs)

        self.eval_data = {
            "Step": [],
            "Mean return": [],
            "Success rate": [],
            "Mean episode length": []
        }

        if self.log_tensorboard:
            self.log_tb = SummaryWriter(str(self.log_dir_path))

        self.n_step_done = n_steps_done

    def reset_all(self):
        self.returns = np.zeros(self.n_parrallel_envs)
        self.success = [False] * self.n_parrallel_envs
        self.ep_lengths = np.zeros(self.n_parrallel_envs)

    def _reset_env(self, env_i):
        self.returns[env_i] = 0.0
        self.success[env_i] = False
        self.ep_lengths[env_i] = 0.0

    def _store_episode(self, env_i):
        self.train_data["Step"].append(self.n_step_done)
        self.train_data["Episode return"].append(self.returns[env_i])
        self.train_data["Success"].append(self.success[env_i])
        self.train_data["Episode length"].append(self.ep_lengths[env_i])

        # Log Tensorboard
        if self.log_tensorboard:
            self.log_tb.add_scalar(
                'agent0/episode_return', 
                self.train_data["Episode return"][-1], 
                self.train_data["Step"][-1])

    def count_returns(self, step, rewards, dones):
        global_rewards = rewards.mean(axis=1)
        global_dones = dones.all(axis=1)
        for e_i in range(self.n_parrallel_envs):
            self.returns[e_i] += global_rewards[e_i]
            self.ep_lengths[e_i] += 1
            if global_dones[e_i]:
                if self.ep_lengths[e_i] < self.max_ep_length:
                    self.success[e_i] = True
                # TODO: fix ça en mettant step dans la classe pour garder un vrai compte du nombre de steps terminées
                self.n_step_done += self.ep_lengths[e_i]
                self._store_episode(e_i)
                self._reset_env(e_i)

    def log_comm(self, step, comm_rewards, losses=None):

        # Log Tensorboard
        if self.log_tensorboard:
            self.log_tb.add_scalars('agent0/comm_reward', comm_rewards, step)
            if losses is not None:
                self.log_tb.add_scalars('agent0/losses', losses, step)

    def log_losses(self, losses, step):
        if self.log_tensorboard:
            if type(losses) is dict:
                pass
            elif type(losses) is tuple:
                pol_losses, lang_losses = losses
                losses = {
                    "value_loss": np.mean(
                        [a_l["value_loss"] for a_l in pol_losses]),
                    "policy_loss": np.mean(
                        [a_l["policy_loss"] for a_l in pol_losses]),
                    "clip_loss": lang_losses[0], 
                    "capt_loss": lang_losses[1], 
                    "mean_sim": lang_losses[2]}
            else:
                losses = {
                    "value_loss": np.mean(
                        [a_l["value_loss"] for a_l in losses]),
                    "policy_loss": np.mean(
                        [a_l["policy_loss"] for a_l in losses])}
            self.log_tb.add_scalars(
                'agent0/losses', losses, step)

    def log_eval(self, step, mean_return, success_rate, mean_ep_len):
        self.eval_data["Step"].append(step)
        self.eval_data["Mean return"].append(mean_return)
        self.eval_data["Success rate"].append(success_rate)
        self.eval_data["Mean episode length"].append(mean_ep_len)

    def save(self):
        train_df = pd.DataFrame(self.train_data)

        # Append to previous data
        train_path = str(self.log_dir_path / 'training_data.csv')
        if os.path.isfile(train_path):
            previous_data = pd.read_csv(train_path)
            train_df = pd.concat([previous_data, train_df])

        train_df.to_csv(train_path)


        if self.do_eval:
            eval_df = pd.DataFrame(self.eval_data)
            # Append to previous data
            eval_path = str(self.log_dir_path / 'evaluation_data.csv')
            if os.path.isfile(eval_path):
                previous_data = pd.read_csv(eval_path)
                eval_df = pd.concat([previous_data, eval_df])
            eval_df.to_csv(str(self.log_dir_path / 'evaluation_data.csv'))

    def save_n_close(self):
        self.save()
        if self.log_tensorboard:
            # self.log_tb.export_scalars_to_json(
            #     str(self.log_dir_path / 'summary.json'))
            self.log_tb.close()