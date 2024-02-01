import torch
import numpy as np
import torch.nn as nn

from .utils import huber_loss
from .valuenorm import ValueNorm

##########################################################################
# Code modified from https://github.com/marlbenchmark/on-policy
##########################################################################


class ACC_MAPPOTrainAlgo:
    """
    Trainer class for MAPPO to update policies.
    :param args: (dict) training arguments.
    :param agents: (ACC_Agent) policy to update.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """
    def __init__(self, args, agents, device=torch.device("cpu")):
        self.agents = agents
        self.device = device

        self.clip_param = args.clip_param
        self.ppo_epoch = args.ppo_epoch
        self.n_mini_batch = args.n_mini_batch
        self.value_loss_coef = args.value_loss_coef
        self.entropy_coef = args.entropy_coef
        self.max_grad_norm = args.max_grad_norm
        self.huber_delta = args.huber_delta
        self.share_params = args.share_params

        # self._use_recurrent_policy = args["use_recurrent_policy"] TRUE
        # self._use_naive_recurrent = args["use_naive_recurrent_policy"] FALSE
        # self._use_max_grad_norm = args["use_max_grad_norm"] TRUE
        # self._use_clipped_value_loss = args["use_clipped_value_loss"] TRUE
        # self._use_huber_loss = args["use_huber_loss"] TRUE
        # self._use_popart = args["use_popart"] FALSE
        # self._use_valuenorm = args["use_valuenorm"] TRUE
        # self._use_value_active_masks = args["use_value_active_masks"] FALSE
        # self._use_policy_active_masks = args["use_policy_active_masks"] FALSE
        
        # if self._use_popart:
        #     self.value_normalizer = self.policy.critic.v_out
        # elif self._use_valuenorm:
        self.value_normalizer = ValueNorm(1).to(device)

    def _compute_policy_loss(self, 
            action_log_probs, old_action_log_probs_batch, adv_targ, dist_entropy):
        imp_weights = torch.exp(action_log_probs - old_action_log_probs_batch)

        surr1 = imp_weights * adv_targ
        surr2 = torch.clamp(
            imp_weights, 
            1.0 - self.clip_param, 
            1.0 + self.clip_param) * adv_targ

        loss = -torch.sum(
            torch.min(surr1, surr2), dim=-1, keepdim=True).mean()
        
        loss = loss - dist_entropy * self.entropy_coef

        return loss

    def _compute_value_loss(self, values, value_preds_batch, return_batch):
        """
        Calculate value function loss.
        :param values: (torch.Tensor) value function predictions.
        :param value_preds_batch: (torch.Tensor) "old" value  predictions from
            data batch (used for value clip loss)
        :param return_batch: (torch.Tensor) reward to go returns.

        :return value_loss: (torch.Tensor) value function loss.
        """
        value_pred_clipped = value_preds_batch + \
            (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
        self.value_normalizer.update(return_batch)
        error_clipped = self.value_normalizer.normalize(
            return_batch) - value_pred_clipped
        error_original = self.value_normalizer.normalize(
            return_batch) - values

        value_loss_clipped = huber_loss(error_clipped, self.huber_delta)
        value_loss_original = huber_loss(error_original, self.huber_delta)

        value_loss = torch.max(value_loss_original, value_loss_clipped)
        
        value_loss = value_loss.mean()

        return value_loss

    def ppo_update(self, policy, sample, train_comm_head=True):
        """
        Update actor and critic networks.
        :param sample: (Tuple) contains data batch with which to update networks.
        :param train_comm_head: (bool) whether to train the communicator head.

        :return value_loss: (torch.Tensor) value function loss.
        :return critic_grad_norm: (torch.Tensor) gradient norm from critic up9date.
        ;return policy_loss: (torch.Tensor) actor(policy) loss value.
        :return dist_entropy: (torch.Tensor) action entropies.
        :return actor_grad_norm: (torch.Tensor) gradient norm from actor update.
        :return imp_weights: (torch.Tensor) importance sampling weights.
        """
        obs_batch, shared_obs_batch, rnn_states_batch, critic_rnn_states_batch, \
            env_actions_batch, comm_actions_batch, \
            old_env_action_log_probs_batch, old_comm_action_log_probs_batch, \
            value_preds_batch, returns_batch, masks_batch, advantages_batch \
                = sample

        obs_batch = torch.from_numpy(obs_batch).to(self.device)
        shared_obs_batch = torch.from_numpy(shared_obs_batch).to(self.device)
        rnn_states_batch = torch.from_numpy(rnn_states_batch).to(self.device)
        critic_rnn_states_batch = torch.from_numpy(critic_rnn_states_batch).to(
            self.device)
        env_actions_batch = torch.from_numpy(env_actions_batch).to(self.device)
        comm_actions_batch = torch.from_numpy(comm_actions_batch).to(self.device)
        value_preds_batch = torch.from_numpy(value_preds_batch).to(self.device)
        returns_batch = torch.from_numpy(returns_batch).to(self.device)
        masks_batch = torch.from_numpy(masks_batch).to(self.device)
        old_env_action_log_probs_batch = torch.from_numpy(
            old_env_action_log_probs_batch).to(self.device)
        old_comm_action_log_probs_batch = torch.from_numpy(
            old_comm_action_log_probs_batch).to(self.device)
        advantages_batch = torch.from_numpy(advantages_batch).to(self.device)

        values, env_action_log_probs, env_dist_entropy, comm_action_log_probs, \
            comm_dist_entropy = policy.evaluate_actions(
                obs_batch, shared_obs_batch, rnn_states_batch, 
                critic_rnn_states_batch, env_actions_batch, comm_actions_batch, 
                masks_batch, train_comm_head)

        # Actor loss
        actor_loss = self._compute_policy_loss(
            env_action_log_probs, 
            old_env_action_log_probs_batch, 
            advantages_batch, 
            env_dist_entropy)

        # Communicator loss
        if train_comm_head:
            comm_loss = self._compute_policy_loss(
                comm_action_log_probs, 
                old_comm_action_log_probs_batch, 
                advantages_batch, 
                comm_dist_entropy)
        else:
            comm_loss = torch.zeros_like(actor_loss)

        # Compute gradients
        policy.act_comm_optim.zero_grad()
        act_comm_loss = actor_loss + comm_loss
        act_comm_loss.backward()

        # Clip gradients
        actor_grad_norm = nn.utils.clip_grad_norm_(
            policy.act_comm.parameters(), self.max_grad_norm)

        # Actor-Communicator update
        policy.act_comm_optim.step()

        # Critic loss
        value_loss = self._compute_value_loss(
            values, value_preds_batch, returns_batch)

        # Compute gradients
        policy.critic_optim.zero_grad()
        (value_loss * self.value_loss_coef).backward()

        # Clip gradients
        critic_grad_norm = nn.utils.clip_grad_norm_(
            policy.critic.parameters(), self.max_grad_norm)

        # Critic update
        policy.critic_optim.step()

        return value_loss, actor_loss, comm_loss

    def train(self, buffer, train_comm_head=True):
        """
        Perform a training update using minibatch GD.
        :param buffer: (SharedReplayBuffer) buffer containing training data.
        :param train_comm_head: (bool) whether to train the communicator head.

        :return losses: (dict) contains information regarding training 
            update (e.g. loss, grad norms, etc).
        """
        # Compute and normalize advantages
        advantages = buffer.returns[:-1] - self.value_normalizer.denormalize(
            buffer.value_preds[:-1])
        advantages_copy = advantages.copy()
        # advantages_copy[buffer.active_masks[:-1] == 0.0] = np.nan
        mean_advantages = np.nanmean(advantages_copy)
        std_advantages = np.nanstd(advantages_copy)
        advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)
        
        losses = {
            "value_loss": 0.0,
            "actor_loss": 0.0}
        if train_comm_head:
            losses["comm_loss"] = 0.0
        if not self.share_params:
            for a_i in range(len(self.agents) - 1):
                losses["value_loss_" + str(a_i + 1)] = 0.0
                losses["actor_loss_" + str(a_i + 1)] = 0.0
                if train_comm_head:
                    losses["comm_loss_" + str(a_i + 1)] = 0.0

        for _ in range(self.ppo_epoch):
            data_generator = buffer.my_recurrent_generator(advantages)
    
            for sample in data_generator:
                if self.share_params:
                    value_loss, actor_loss, comm_loss = self.ppo_update(
                        self.agents[0], sample, train_comm_head)

                    losses["value_loss"] += value_loss.item()
                    losses["actor_loss"] += actor_loss.item()
                    if train_comm_head:
                        losses["comm_loss"] += comm_loss.item()
                else:
                    for a_i in range(len(self.agents)):
                        sample_i = tuple([batch[:, a_i] for batch in sample])

                        value_loss, actor_loss, comm_loss = self.ppo_update(
                            self.agents[a_i], sample_i, train_comm_head)

                        a_name = "_" + str(a_i) if a_i > 0 else ""
                        losses["value_loss" + a_name] += value_loss.item()
                        losses["actor_loss" + a_name] += actor_loss.item()
                        if train_comm_head:
                            losses["comm_loss" + a_name] += comm_loss.item()

        num_updates = self.ppo_epoch * self.n_mini_batch
        for k in losses.keys():
            losses[k] /= num_updates
 
        return losses