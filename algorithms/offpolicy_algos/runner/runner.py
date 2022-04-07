from abc import ABC, abstractmethod


class Runner(ABC):
    """
    Abstract runner class. Runs rollouts for either training or evaluation 
    episodes.
    """
    @abstractmethod
    def train_rollout(self, ep_i):
        """
        Rollouts a training episode.
        :param ep_i: training iteration
        """
        pass

    @abstractmethod
    def eval_rollout(self):
        pass