from tensorboardX import SummaryWriter


class Logger():
    """
    Class for our logger that saves training and evaluation data during 
    training.
    :param log_dir_path: (str) path of directory where logs are saved
    :param do_tensorboard: (bool) whether to log in tensorboard
    """
    def __init__(self, log_dir_path, do_tensorboard=True):
        self.log_dir_path = log_dir_path
        self.do_tensorboard = do_tensorboard

        self.train_data = {
            "Step": [],
            "Episode return": [],
            "Success": [],
            "Episode length": []
        }

        self.eval_data = {
            "Step": [],
            "Mean return": [],
            "Success rate": [],
            "Mean episode length": []
        }

        if self.do_tensorboard:
            self.log_tb = SummaryWriter(str(log_dir))

    def init_train_data(self):
        self.training_data = {
            "Step": [],
            "Episode return": [],
            "Success": [],
            "Episode length": []
        }

    def init_eval_data(self):
        self.eval_data = {
            "Step": [],
            "Mean return": [],
            "Success rate": [],
            "Mean episode length": []
        }

    def log(self, rewards, dones):
        """
        Log training data.
        :param rewards: (numpy.ndarray) List of rewards in multiple parrallel
            environments
        :param dones: (numpy.ndarray) List of dones in multiple parrallel
            environments

        :output tot_steps: (int) number of steps performed in all parallel 
            environments
        """
        train_data_dict["Step"].append(step_i)
        train_data_dict["Episode return"].append(
            np.sum(ep_rews) / nb_agents)
        train_data_dict["Episode extrinsic return"].append(
            np.mean(ep_ext_returns))
        train_data_dict["Episode intrinsic return"].append(
            np.mean(ep_int_returns))
        train_data_dict["Success"].append(int(ep_success))
        train_data_dict["Episode length"].append(ep_step_i + 1)
        # Log Tensorboard
        logger.add_scalar(
            'agent0/episode_return', 
            train_data_dict["Episode return"][-1], 
            train_data_dict["Step"][-1])
        logger.add_scalar(
            'agent0/episode_ext_return', 
            train_data_dict["Episode extrinsic return"][-1], 
            train_data_dict["Step"][-1])
        logger.add_scalar(
            'agent0/episode_int_return', 
            train_data_dict["Episode intrinsic return"][-1], 
            train_data_dict["Step"][-1])