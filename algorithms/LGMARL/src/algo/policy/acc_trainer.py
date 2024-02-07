import torch
import numpy as np
import torch.nn as nn

from .utils import huber_loss
from .valuenorm import ValueNorm


class ACC_Trainer:

    def __init__(self, args, agents, lang_learner, device=torch.device("cpu")):
        self.agents = agents
        self.lang_learner = lang_learner
        self.device = device

        self.clip_param = args.clip_param
        self.ppo_epoch = args.ppo_epoch
        self.n_mini_batch = args.n_mini_batch
        self.value_loss_coef = args.value_loss_coef
        self.entropy_coef = args.entropy_coef
        self.max_grad_norm = args.max_grad_norm
        self.huber_delta = args.huber_delta
        self.share_params = args.share_params

        self.value_normalizer = ValueNorm(1).to(device)

    def _compute_advantages(self, buffer):
         # Compute and normalize advantages
        advantages = buffer.returns[:-1] - self.value_normalizer.denormalize(
            buffer.value_preds[:-1])
        advantages_copy = advantages.copy()
        # advantages_copy[buffer.active_masks[:-1] == 0.0] = np.nan
        mean_advantages = np.nanmean(advantages_copy)
        std_advantages = np.nanstd(advantages_copy)
        advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)
        return advantages

    def _update(self, agent, sample, train_comm_head, train_lang):
        # Agent forward pass

        # Language forward pass

        # Actor loss

        # Communicator loss

        # Value loss

        # CLIP loss

        # Captioning loss
        pass

    def train(self, buffer, 
            warmup=False, train_comm_head=True, train_lang=True):
        """
        Perform a training update using minibatch GD.
        :param buffer: (SharedReplayBuffer) buffer containing training data.
        :param train_comm_head: (bool) whether to train the communicator head.

        :return losses: (dict) contains information regarding training 
            update (e.g. loss, grad norms, etc).
        """
        for a in self.agents:
            a.warmup_lr(warmup)
            
        advantages = self._compute_advantages(buffer)
        
        losses = {
            "value_loss": 0.0,
            "actor_loss": 0.0}
        if train_comm_head:
            losses["comm_loss"] = 0.0
        # if not self.share_params:
        #     for a_i in range(len(self.agents) - 1):
        #         losses["value_loss_" + str(a_i + 1)] = 0.0
        #         losses["actor_loss_" + str(a_i + 1)] = 0.0
        #         if train_comm_head:
        #             losses["comm_loss_" + str(a_i + 1)] = 0.0

        for _ in range(self.ppo_epoch):
            data_generator = buffer.my_recurrent_generator(advantages)
    
            for sample in data_generator:
                if self.share_params:
                    self._update(self.agents[0], sample, train_comm_head, train_lang)
                    # value_loss, actor_loss, comm_loss = self.ppo_update(
                    #     self.agents[0], sample, train_comm_head)

                    # losses["value_loss"] += value_loss.item()
                    # losses["actor_loss"] += actor_loss.item()
                    # if train_comm_head:
                    #     losses["comm_loss"] += comm_loss.item()
                else:
                    for a_i in range(len(self.agents)):
                        sample_i = tuple([batch[:, a_i] for batch in sample])

                        # value_loss, actor_loss, comm_loss = self.ppo_update(
                        #     self.agents[a_i], sample_i, train_comm_head)

                        # # a_name = "_" + str(a_i) if a_i > 0 else ""
                        # losses["value_loss"] += value_loss.item()
                        # losses["actor_loss"] += actor_loss.item()
                        # if train_comm_head:
                        #     losses["comm_loss"] += comm_loss.item()

        num_updates = self.ppo_epoch * self.n_mini_batch
        for k in losses.keys():
            losses[k] /= num_updates
            if not self.share_params:
                losses[k] /= len(self.agents)
 
        return losses