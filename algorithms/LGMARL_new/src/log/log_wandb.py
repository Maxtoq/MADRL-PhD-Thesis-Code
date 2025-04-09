import os
import numpy as np
import pandas as pd
import wandb

# from requests.exceptions import RequestException

# def safe_wandb_init(**kwargs):
#     try:
#         return wandb.init(**kwargs)
#     except wandb.errors.errors.CommError:
#         wandb.finish()
#         print("WANDB connection timed out — switching to offline mode.")
#         os.environ["WANDB_MODE"] = "offline"
#         return wandb.init(**kwargs)


class Logger():
    """
    W&B Logger for saving training and evaluation data during training.
    :param args: (argparse.Namespace) training arguments
    :param log_dir_path: (str) path of directory where logs are saved
    """
    def __init__(self, args, log_dir_path, n_steps_done=0):
        self.log_dir_path = log_dir_path
        self.max_ep_length = args.episode_length
        self.log_wandb = args.log_tensorboard  # Same flag reused for wandb
        self.n_parallel_envs = args.n_parallel_envs
        self.do_eval = args.do_eval

        self.train_data = {
            "Step": [],
            "Episode return": [],
            "Success": [],
            "Episode length": []
        }
        self.returns = np.zeros(self.n_parallel_envs)
        self.success = [False] * self.n_parallel_envs
        self.ep_lengths = np.zeros(self.n_parallel_envs)

        self.ratio_gen_perf = {
            "Step": [],
            "Ratio_gen_perf": []
        }

        self.eval_data = {
            "Step": [],
            "Mean return": [],
            "Success rate": [],
            "Mean episode length": []
        }

        self.n_step_done = n_steps_done

        if self.log_wandb:
            wandb.init(
                project=f"{args.env_name}-{args.experiment_name}",  # Replace with your project name
                config={**vars(args)},
                dir="outputs")
            
    def reset_all(self):
        self.returns = np.zeros(self.n_parallel_envs)
        self.success = [False] * self.n_parallel_envs
        self.ep_lengths = np.zeros(self.n_parallel_envs)

    def _reset_env(self, env_i):
        self.returns[env_i] = 0.0
        self.success[env_i] = False
        self.ep_lengths[env_i] = 0.0

    def _store_episode(self, env_i):
        self.train_data["Step"].append(self.n_step_done)
        self.train_data["Episode return"].append(self.returns[env_i])
        self.train_data["Success"].append(self.success[env_i])
        self.train_data["Episode length"].append(self.ep_lengths[env_i])

        if self.log_wandb:
            wandb.log({
                "train/episode_return": self.returns[env_i],
                "train/success": int(self.success[env_i]),
                "train/episode_length": self.ep_lengths[env_i],
                "train/step": self.n_step_done
            })

    def count_returns(self, step, rewards, dones):
        global_rewards = rewards.mean(axis=1)
        global_dones = dones.all(axis=1)
        for e_i in range(self.n_parallel_envs):
            self.returns[e_i] += global_rewards[e_i]
            self.ep_lengths[e_i] += 1
            if global_dones[e_i]:
                if self.ep_lengths[e_i] < self.max_ep_length:
                    self.success[e_i] = True
                self.n_step_done += self.ep_lengths[e_i]
                self._store_episode(e_i)
                self._reset_env(e_i)

    def log_comm(self, step, comm_rewards, losses=None):
        if self.log_wandb:
            log_dict = {"comm/step": step}
            log_dict.update({f"comm_reward/{k}": v for k, v in comm_rewards.items()})
            if losses is not None:
                log_dict.update({f"loss/{k}": v for k, v in losses.items()})
            wandb.log(log_dict)

    def log_losses(self, losses, step):
        if self.log_wandb:
            log_dict = {"loss/step": step}

            if isinstance(losses, dict):
                log_dict.update({f"loss/{k}": v for k, v in losses.items()})
            elif isinstance(losses, tuple):
                pol_losses, lang_losses = losses
                log_dict.update({
                    "loss/value_loss": np.mean([l["value_loss"] for l in pol_losses]),
                    "loss/policy_loss": np.mean([l["policy_loss"] for l in pol_losses]),
                    "loss/clip_loss": lang_losses[0],
                    "loss/capt_loss": lang_losses[1],
                    "loss/mean_sim": lang_losses[2],
                })
            else:
                log_dict.update({
                    "loss/value_loss": np.mean([l["value_loss"] for l in losses]),
                    "loss/policy_loss": np.mean([l["policy_loss"] for l in losses]),
                })

            wandb.log(log_dict)

        if "ratio_gen_perf" in losses:
            self.ratio_gen_perf["Step"].append(step)
            self.ratio_gen_perf["Ratio_gen_perf"].append(losses["ratio_gen_perf"])

    def log_eval(self, step, mean_return, success_rate, mean_ep_len):
        self.eval_data["Step"].append(step)
        self.eval_data["Mean return"].append(mean_return)
        self.eval_data["Success rate"].append(success_rate)
        self.eval_data["Mean episode length"].append(mean_ep_len)

        if self.log_wandb:
            wandb.log({
                "eval/mean_return": mean_return,
                "eval/success_rate": success_rate,
                "eval/mean_ep_len": mean_ep_len,
                "eval/step": step})

    def save(self):
        train_df = pd.DataFrame(self.train_data)
        train_path = str(self.log_dir_path / 'training_data.csv')
        if os.path.isfile(train_path):
            previous_data = pd.read_csv(train_path)
            train_df = pd.concat([previous_data, train_df], ignore_index=True)
        train_df.to_csv(train_path)

        if len(self.ratio_gen_perf["Step"]) > 0:
            rgp_df = pd.DataFrame(self.ratio_gen_perf)
            rgp_path = str(self.log_dir_path / 'comm_data.csv')
            if os.path.isfile(rgp_path):
                previous_data = pd.read_csv(rgp_path, index_col=0)
                rgp_df = pd.concat([previous_data, rgp_df], ignore_index=True)
            rgp_df.to_csv(rgp_path)

        if self.do_eval:
            eval_df = pd.DataFrame(self.eval_data)
            eval_path = str(self.log_dir_path / 'evaluation_data.csv')
            if os.path.isfile(eval_path):
                previous_data = pd.read_csv(eval_path)
                eval_df = pd.concat([previous_data, eval_df])
            eval_df.to_csv(eval_path)

    def save_n_close(self):
        self.save()
        if self.log_wandb:
            wandb.finish()