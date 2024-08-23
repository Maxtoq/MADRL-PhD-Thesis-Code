import torch
import numpy as np
import torch.nn as nn

from .utils import huber_loss
from .valuenorm import ValueNorm


class Trainer:

    def __init__(self, args, model, buffer, device=torch.device("cpu")):
        self.model = model
        self.agents = model.agents if type(model.agents) == list else [model.agents]
        self.buffer = buffer
        self.device = device
        self.share_params = args.share_params

        # PPO params
        self.clip_param = args.clip_param
        self.ppo_epoch = args.ppo_epoch
        self.n_mini_batch = args.n_mini_batch
        self.value_loss_coef = args.value_loss_coef
        self.entropy_coef = args.entropy_coef
        self.max_grad_norm = args.max_grad_norm
        self.huber_delta = args.huber_delta

        # Language params
        self.lang_batch_size = args.lang_batch_size
        self.lang_imp_sample = args.lang_imp_sample
        self.temp = args.lang_temp

        # Init loss weights
        self.dyna_weight_loss = args.dyna_weight_loss
        self.capt_loss_w = args.lang_capt_loss_weight
        self.actor_loss_w = args.actor_loss_weight
        self.comm_loss_w = 1.0
        self.act_value_loss_w =1.0
        self.comm_value_loss_w = 1.0
        self.clip_loss_w = 1.0

        self.clip_loss = nn.CrossEntropyLoss()
        self.captioning_loss = nn.NLLLoss(reduction="mean", ignore_index=0)

        self.env_value_normalizer = ValueNorm(1, norm_axes=2).to(device)
        self.comm_value_normalizer = ValueNorm(1, norm_axes=2).to(device)

    def _update_loss_weights(self, losses):
        self.actor_loss_w = 1 / (
            abs(losses["actor_loss"]) 
            if losses["actor_loss"] != 0 else self.actor_loss_w)
        self.act_value_loss_w = 1 / (
            abs(losses["env_value_loss"]) 
            if losses["env_value_loss"] != 0 else self.act_value_loss_w)
        if "comm_loss" in losses:
            self.comm_loss_w = 1 / (
                abs(losses["comm_loss"]) 
                if losses["comm_loss"] != 0 else self.comm_loss_w)
            # self.comm_loss_w[a_i] = min(self.comm_loss_w[a_i], new_weight)
            self.comm_value_loss_w = 1 / (
                abs(losses["comm_value_loss"]) 
                if losses["comm_value_loss"] != 0 else self.comm_value_loss_w)
        if "clip_loss" in losses:
            self.clip_loss_w = 1 / (
                abs(losses["clip_loss"]) 
                if losses["clip_loss"] != 0 else self.clip_loss_w)
            self.capt_loss_w = 1 / (
                abs(losses["capt_loss"]) 
                if losses["capt_loss"] != 0 else self.capt_loss_w)

    def _compute_advantages(self, train_comm_head):
         # Compute and normalize action advantages
        act_advantages = self.buffer.act_returns[:-1] \
            - self.env_value_normalizer.denormalize(
                self.buffer.act_value_preds[:-1])
        act_advantages_copy = act_advantages.copy()
        mean_act_advantages = np.nanmean(act_advantages_copy)
        std_act_advantages = np.nanstd(act_advantages_copy)
        act_advantages = (act_advantages - mean_act_advantages) \
            / (std_act_advantages + 1e-5)
         # Compute and normalize communication advantages
        if train_comm_head:
            comm_advantages = self.buffer.comm_returns[:-1] \
                - self.comm_value_normalizer.denormalize(
                    self.buffer.comm_value_preds[:-1])
            comm_advantages_copy = comm_advantages.copy()
            mean_comm_advantages = np.nanmean(comm_advantages_copy)
            std_comm_advantages = np.nanstd(comm_advantages_copy)
            comm_advantages = (comm_advantages - mean_comm_advantages) \
                / (std_comm_advantages + 1e-5)
        else:
            comm_advantages = np.zeros_like(act_advantages)
        return act_advantages, comm_advantages

    def _compute_policy_loss(self, 
            action_log_probs, old_action_log_probs_batch, adv_targ, dist_entropy):
        ratio = torch.exp(action_log_probs - old_action_log_probs_batch)

        surr1 = ratio * adv_targ
        surr2 = torch.clamp(
            ratio, 
            1.0 - self.clip_param, 
            1.0 + self.clip_param) * adv_targ

        loss = -torch.min(surr1, surr2).mean(dim=0)
        
        loss = (loss - dist_entropy * self.entropy_coef).mean()

        return loss

    def _compute_value_loss(self, 
            values, value_preds_batch, return_batch, value_norm):
        """
        Calculate value function loss.
        :param values: (torch.Tensor) value function predictions.
        :param value_preds_batch: (torch.Tensor) "old" value  predictions from
            data batch (used for value clip loss)
        :param return_batch: (torch.Tensor) reward to go returns.
        :param value_norm: (ValueNorm) value normalizer instance.

        :return value_loss: (torch.Tensor) value function loss.
        """
        value_pred_clipped = value_preds_batch + \
            (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
        value_norm.update(return_batch)
        error_clipped = value_norm.normalize(return_batch) - value_pred_clipped
        error_original = value_norm.normalize(return_batch) - values

        value_loss_clipped = huber_loss(error_clipped, self.huber_delta)
        value_loss_original = huber_loss(error_original, self.huber_delta)

        value_loss = torch.max(value_loss_original, value_loss_clipped)
        
        value_loss = value_loss.mean()

        return value_loss

    def _compute_clip_loss(self, obs_contexts, lang_contexts):
        # Compute similarity
        norm_obs_contexts = obs_contexts / obs_contexts.norm(
            dim=1, keepdim=True)
        norm_lang_contexts = lang_contexts / lang_contexts.norm(
            dim=1, keepdim=True)
        sim = norm_obs_contexts @ norm_lang_contexts.t() * self.temp
        mean_sim = sim.diag().mean()

        # Compute CLIP loss
        labels = torch.arange(obs_contexts.shape[0]).to(self.device)
        loss_o = self.clip_loss(sim, labels)
        loss_l = self.clip_loss(sim.t(), labels)
        clip_loss = (loss_o + loss_l) / 2

        return clip_loss, mean_sim.item()

    def _compute_capt_loss(self, preds, targets):
        dec_loss = 0
        preds = preds.reshape(-1, preds.shape[-1])
        targets = targets.reshape(-1)
        capt_loss = self.captioning_loss(preds, targets)
        return capt_loss

    # def _train_mappo(self, agent_i, sample, train_comm_head, train_lang):
    #     obs_batch, joint_obs_batch, obs_enc_rnn_states_batch, \
    #         joint_obs_enc_rnn_states_batch, comm_enc_rnn_states_batch, \
    #         env_actions_batch, comm_actions_batch, \
    #         old_env_action_log_probs_batch, old_comm_action_log_probs_batch, \
    #         act_value_preds_batch, comm_value_preds_batch, act_returns_batch, \
    #         comm_returns_batch, masks_batch, act_advt_batch, comm_advt_batch, \
    #         gen_comm_batch, perf_messages_batch, perf_broadcasts_batch = sample

    #     obs_batch = torch.from_numpy(obs_batch).to(self.device)
    #     joint_obs_batch = torch.from_numpy(joint_obs_batch).to(self.device)
    #     obs_enc_rnn_states_batch = torch.from_numpy(
    #         obs_enc_rnn_states_batch).to(self.device)
    #     joint_obs_enc_rnn_states_batch = torch.from_numpy(
    #         joint_obs_enc_rnn_states_batch).to(self.device)
    #     comm_enc_rnn_states_batch = torch.from_numpy(
    #         comm_enc_rnn_states_batch).to(self.device)
    #     env_actions_batch = torch.from_numpy(env_actions_batch).to(self.device)
    #     comm_actions_batch = torch.from_numpy(comm_actions_batch).to(self.device)
    #     act_value_preds_batch = torch.from_numpy(
    #         act_value_preds_batch).to(self.device)
    #     comm_value_preds_batch = torch.from_numpy(
    #         comm_value_preds_batch).to(self.device)
    #     act_returns_batch = torch.from_numpy(act_returns_batch).to(self.device)
    #     comm_returns_batch = torch.from_numpy(comm_returns_batch).to(self.device)
    #     masks_batch = torch.from_numpy(masks_batch).to(self.device)
    #     old_env_action_log_probs_batch = torch.from_numpy(
    #         old_env_action_log_probs_batch).to(self.device)
    #     old_comm_action_log_probs_batch = torch.from_numpy(
    #         old_comm_action_log_probs_batch).to(self.device)
    #     act_advt_batch = torch.from_numpy(act_advt_batch).to(self.device)
    #     comm_advt_batch = torch.from_numpy(comm_advt_batch).to(self.device)
    #     perf_messages_batch = torch.from_numpy(perf_messages_batch).to(self.device)

    #     # Agent forward pass
    #     act_values, comm_values, env_action_log_probs, act_dist_entropy, \
    #     comm_action_log_probs, comm_dist_entropy, comm_actions \
    #         = self.agents[agent_i].evaluate_actions(
    #             obs_batch, joint_obs_batch, obs_enc_rnn_states_batch, 
    #             joint_obs_enc_rnn_states_batch, comm_enc_rnn_states_batch, 
    #             env_actions_batch, comm_actions_batch, masks_batch)

    #     # Actor loss
    #     actor_loss = self._compute_policy_loss(
    #         env_action_log_probs, 
    #         old_env_action_log_probs_batch, 
    #         act_advt_batch, 
    #         act_dist_entropy)
    #     # Act Value loss
    #     act_value_loss = self._compute_value_loss(
    #         act_values, 
    #         act_value_preds_batch, 
    #         act_returns_batch, 
    #         self.env_value_normalizer)

    #     log_losses = {
    #         "actor_loss": actor_loss.item(),
    #         "env_value_loss": act_value_loss.item()}

    #     # Communicator losses
    #     if train_comm_head:
    #         # The comm head is trained only on generated messages
    #         gen_comm_batch = gen_comm_batch == 1.0
    #         if gen_comm_batch.sum() > 32:
    #             comm_loss = self._compute_policy_loss(
    #                 comm_action_log_probs[gen_comm_batch], 
    #                 old_comm_action_log_probs_batch[gen_comm_batch], 
    #                 comm_advt_batch[gen_comm_batch], 
    #                 comm_dist_entropy)
    #             log_losses["comm_loss"] = comm_loss.item()
    #         else:
    #             comm_loss = torch.zeros_like(actor_loss)

    #         comm_value_loss = self._compute_value_loss(
    #             comm_values, 
    #             comm_value_preds_batch, 
    #             comm_returns_batch, 
    #             self.comm_value_normalizer)
            
    #         log_losses["comm_value_loss"] = comm_value_loss.item()
    #     else:
    #         comm_loss = torch.zeros_like(actor_loss)
    #         comm_value_loss = torch.zeros_like(act_value_loss)

    #     # Language losses
    #     if train_lang:
    #         # Sample a mini-batch
    #         batch_size = min(len(perf_broadcasts_batch), self.lang_batch_size)
    #         ids = np.random.choice(
    #             len(perf_broadcasts_batch), 
    #             size=batch_size, 
    #             replace=False,
    #             p=mess_sampling_probs if self.lang_imp_sample else None)

    #         # CLIP loss 
    #         # Encode sentences
    #         sample_broadcasts = [perf_broadcasts_batch[i] for i in ids]
    #         lang_contexts = self.lang_learner.lang_encoder(sample_broadcasts)
    #         lang_contexts = lang_contexts.squeeze()
    #         obs_contexts = critic_obs_encs[ids]
    #         # CLIP loss
    #         clip_loss, mean_sim = self._compute_clip_loss(
    #             obs_contexts, lang_contexts)

    #         log_losses["clip_loss"] = clip_loss.item()
    #         log_losses["mean_sim"] = mean_sim

    #         # Captioning loss
    #         # sample_perf_messages = [perf_messages_batch[i] for i in ids]

    #         dec_inputs = comm_actions[ids]
    #         # Decode
    #         targets = perf_messages_batch[ids].long()
    #         # Remove excess padded tokens
    #         n_excess = min((targets == 0).sum(-1))
    #         if n_excess > 0:
    #             targets = targets[:, :-min((targets == 0).sum(-1))]
    #         # encoded_targets = self.lang_learner.word_encoder.encode_batch(
    #         #     sample_perf_messages, pad=True).to(self.device)
    #         decoder_outputs, _ = self.lang_learner.decoder(
    #             dec_inputs, targets)
    #         # Captioning loss
    #         capt_loss = self._compute_capt_loss(decoder_outputs, targets)

    #         log_losses["capt_loss"] = capt_loss.item()
    #     else:
    #         clip_loss = torch.zeros_like(act_value_loss)
    #         capt_loss = torch.zeros_like(act_value_loss)

    #     if self.dyna_weight_loss:
    #         self._update_loss_weights(log_losses)

    #     loss = self.actor_loss_w * actor_loss \
    #             + 1.0 * self.comm_loss_w * comm_loss \
    #             + self.act_value_loss_w * act_value_loss \
    #             + self.comm_value_loss_w * comm_value_loss \
    #             + self.clip_loss_w * clip_loss \
    #             + self.capt_loss_w * capt_loss
        
    #     # Compute gradients
    #     self.agents[agent_i].optim.zero_grad()
    #     # self.agents[agent_i].critic_optim.zero_grad()
    #     if train_lang:
    #         self.lang_learner.optim.zero_grad()
    #     loss.backward()

    #     # Clip gradients
    #     actcomm_grad_norm = nn.utils.clip_grad_norm_(
    #         self.agents[agent_i].parameters(), self.max_grad_norm)
    #     # critic_grad_norm = nn.utils.clip_grad_norm_(
    #     #     self.agents[agent_i].critic.parameters(), self.max_grad_norm)

    #     # Update
    #     self.agents[agent_i].optim.step()
    #     # self.agents[agent_i].critic_optim.step()
    #     if train_lang:
    #         self.lang_learner.optim.step()

    #     return log_losses

    def _train_mappo_diff(self, sample, train_comm_head, train_lang):
        obs_batch, joint_obs_batch, obs_enc_rnn_states_batch, \
            joint_obs_enc_rnn_states_batch, comm_enc_rnn_states_batch, \
            env_actions_batch, comm_actions_batch, \
            old_env_action_log_probs_batch, old_comm_action_log_probs_batch, \
            env_value_preds_batch, comm_value_preds_batch, env_returns_batch, \
            comm_returns_batch, masks_batch, env_advt_batch, comm_advt_batch, \
            gen_comm_batch, perf_messages_batch, perf_broadcasts_batch = sample

        # obs_batch = torch.from_numpy(obs_batch).to(self.device)
        # joint_obs_batch = torch.from_numpy(joint_obs_batch).to(self.device)
        # obs_enc_rnn_states_batch = torch.from_numpy(
        #     obs_enc_rnn_states_batch).to(self.device)
        # joint_obs_enc_rnn_states_batch = torch.from_numpy(
        #     joint_obs_enc_rnn_states_batch).to(self.device)
        # comm_enc_rnn_states_batch = torch.from_numpy(
        #     comm_enc_rnn_states_batch).to(self.device)
        # env_actions_batch = torch.from_numpy(env_actions_batch).to(self.device)
        # comm_actions_batch = torch.from_numpy(comm_actions_batch).to(self.device)
        env_value_preds_batch = torch.from_numpy(
            env_value_preds_batch).to(self.device)
        comm_value_preds_batch = torch.from_numpy(
            comm_value_preds_batch).to(self.device)
        env_returns_batch = torch.from_numpy(env_returns_batch).to(self.device)
        comm_returns_batch = torch.from_numpy(comm_returns_batch).to(self.device)
        # masks_batch = torch.from_numpy(masks_batch).to(self.device)
        old_env_action_log_probs_batch = torch.from_numpy(
            old_env_action_log_probs_batch).to(self.device)
        old_comm_action_log_probs_batch = torch.from_numpy(
            old_comm_action_log_probs_batch).to(self.device)
        env_advt_batch = torch.from_numpy(env_advt_batch).to(self.device)
        comm_advt_batch = torch.from_numpy(comm_advt_batch).to(self.device)
        # perf_messages_batch = torch.from_numpy(perf_messages_batch).to(self.device)
        
        # Forward all agents
        actions, action_log_probs, env_values, comm_actions, \
            comm_action_log_probs, comm_values, new_obs_rnn_states, \
            new_joint_obs_rnn_states, new_comm_rnn_states, messages, \
            env_action_log_probs, env_dist_entropy, \
            comm_action_log_probs, comm_dist_entropy, lang_obs_enc \
            = self.model.comm_n_act(
                obs_batch, joint_obs_batch, obs_enc_rnn_states_batch, 
                joint_obs_enc_rnn_states_batch, comm_enc_rnn_states_batch, 
                masks_batch, perf_messages_batch, perf_broadcasts_batch, 
                deterministic=True, eval_actions=env_actions_batch, 
                eval_comm_actions=comm_actions_batch)

        # Actor loss
        actor_loss = self._compute_policy_loss(
            env_action_log_probs, 
            old_env_action_log_probs_batch, 
            env_advt_batch, 
            env_dist_entropy)

        # Env Value loss
        env_value_loss = self._compute_value_loss(
            env_values, 
            env_value_preds_batch, 
            env_returns_batch, 
            self.env_value_normalizer)

        log_losses = {
            "actor_loss": actor_loss.item(),
            "env_value_loss": env_value_loss.item()}
        
        # Communicator losses
        if train_comm_head:
            # The comm head is trained only on generated messages
            gen_comm_batch = gen_comm_batch == 1.0
            if gen_comm_batch.sum() > 32:
                comm_loss = self._compute_policy_loss(
                    comm_action_log_probs[gen_comm_batch], 
                    old_comm_action_log_probs_batch[gen_comm_batch], 
                    comm_advt_batch[gen_comm_batch], 
                    comm_dist_entropy)
                log_losses["comm_loss"] = comm_loss.item()
            else:
                comm_loss = torch.zeros_like(actor_loss)

            comm_value_loss = self._compute_value_loss(
                comm_values, 
                comm_value_preds_batch, 
                comm_returns_batch, 
                self.comm_value_normalizer)
            
            log_losses["comm_value_loss"] = comm_value_loss.item()
        else:
            comm_loss = torch.zeros_like(actor_loss)
            comm_value_loss = torch.zeros_like(env_value_loss)

        # Language losses
        if train_lang:
            # Sample a mini-batch
            batch_size = min(
                len(perf_broadcasts_batch) * self.model.n_agents, 
                self.lang_batch_size)
            ids = np.random.choice(
                len(perf_broadcasts_batch) * self.model.n_agents, 
                size=batch_size, 
                replace=False)
                # p=mess_sampling_probs if self.lang_imp_sample else None)

            # CLIP loss 
            # Encode sentences
            sample_obs_contexts = lang_obs_enc.reshape(
                len(perf_broadcasts_batch) * self.model.n_agents, -1)[ids]
            sample_broadcasts = [
                perf_broadcasts_batch[i // self.model.n_agents] for i in ids]
            sample_lang_contexts = self.model.lang_learner.encode_sentences(
                sample_broadcasts)
            # CLIP loss
            clip_loss, mean_sim = self._compute_clip_loss(
                sample_obs_contexts, sample_lang_contexts)

            log_losses["clip_loss"] = clip_loss.item()
            log_losses["mean_sim"] = mean_sim

            # Captioning loss
            # Decode
            dec_inputs = comm_actions.reshape(
                len(perf_broadcasts_batch) * self.model.n_agents, -1)[ids]
            targets = torch.from_numpy(
                perf_messages_batch.reshape(
                len(perf_broadcasts_batch) * self.model.n_agents, -1)[ids]
                ).long().to(self.device)
            # Remove excess padded tokens
            n_excess = min((targets == 0).sum(-1))
            if n_excess > 0:
                targets = targets[:, :-min((targets == 0).sum(-1))]
            # encoded_targets = self.lang_learner.word_encoder.encode_batch(
            #     sample_perf_messages, pad=True).to(self.device)
            decoder_outputs, _ = self.model.lang_learner.decoder(
                dec_inputs, targets)
            # Captioning loss
            capt_loss = self._compute_capt_loss(
                decoder_outputs, targets)

            log_losses["capt_loss"] = capt_loss.item()
        else:
            clip_loss = torch.zeros_like(env_value_loss)
            capt_loss = torch.zeros_like(env_value_loss)

        if self.dyna_weight_loss:
            self._update_loss_weights(log_losses)

        loss = self.actor_loss_w * actor_loss \
                + self.comm_loss_w * comm_loss \
                + self.act_value_loss_w * env_value_loss \
                + self.comm_value_loss_w * comm_value_loss \
                + self.clip_loss_w * clip_loss \
                + self.capt_loss_w * capt_loss

        # Compute gradients
        for a in self.agents:
            a.actor_optim.zero_grad()
            a.critic_optim.zero_grad()
        # self.agents[agent_i].critic_optim.zero_grad()
        if train_lang:
            self.model.lang_learner.optim.zero_grad()
        loss.backward()

        # Clip gradients
        for a in self.agents:
            grad_norm = nn.utils.clip_grad_norm_(
                a.parameters(), self.max_grad_norm)
        # critic_grad_norm = nn.utils.clip_grad_norm_(
        #     self.agents[agent_i].critic.parameters(), self.max_grad_norm)

        # Update
        for a in self.agents:
            a.actor_optim.step()
            a.critic_optim.step()
        # self.agents[agent_i].critic_optim.step()
        if train_lang:
            self.model.lang_learner.optim.step()

        return log_losses

    def train_diff(self, warmup=False, train_comm_head=True, train_lang=True):
        """
        Train LGMARL.

        :param train_comm_head: (bool) Whether to train the communicator head.
        :param train_lang: (bool) Whether to train language modules.

        :return losses: (dict) Contains losses obtained during update.
        """
        for a in self.agents:
            a.warmup_lr(warmup)
            
        act_advantages, comm_advantages = self._compute_advantages(
            train_comm_head)
        
        losses = {
            "env_value_loss": 0.0,
            "actor_loss": 0.0}
        if train_comm_head:
            losses["comm_value_loss"] = 0.0
            losses["comm_loss"] = 0.0
        if train_lang:
            losses["clip_loss"] = 0.0
            losses["mean_sim"] = 0.0
            losses["capt_loss"] = 0.0

        # Train policy
        num_updates = self.ppo_epoch * self.n_mini_batch
        for _ in range(self.ppo_epoch):
            data_generator = self.buffer.recurrent_policy_generator(
                act_advantages, comm_advantages)
    
            for sample in data_generator:
                # if self.share_params:
                #     loss = self._train_mappo(
                #         0, sample, train_comm_head, train_lang)
                    
                #     for key in loss:
                #         losses[key] += loss[key] / num_updates
                # else:
                #     for a_i in range(len(self.agents)):
                #         sample_i = (
                #             *[batch[:, a_i] for batch in sample[:-1]],
                #             [s[a_i] for s in sample[-1]])
                #             # sample[-1][a_i])

                #         loss = self._train_mappo(
                #             a_i, 
                #             sample_i, 
                #             train_comm_head, 
                #             train_lang)
                        
                #         for key in loss:
                #             losses[key] += loss[key] / (
                #                 num_updates * len(self.agents))
                losses = self._train_mappo_diff(
                    sample, train_comm_head, train_lang)

        return losses

    # def train(self, warmup=False, train_comm_head=True, train_lang=True):
    #     """
    #     Train LGMARL.

    #     :param train_comm_head: (bool) Whether to train the communicator head.
    #     :param train_lang: (bool) Whether to train language modules.

    #     :return losses: (dict) Contains losses obtained during update.
    #     """
    #     for a in self.agents:
    #         a.warmup_lr(warmup)
            
    #     act_advantages, comm_advantages = self._compute_advantages(
    #         train_comm_head)
        
    #     losses = {
    #         "env_value_loss": 0.0,
    #         "actor_loss": 0.0}
    #     if train_comm_head:
    #         losses["comm_value_loss"] = 0.0
    #         losses["comm_loss"] = 0.0
    #     if train_lang:
    #         losses["clip_loss"] = 0.0
    #         losses["mean_sim"] = 0.0
    #         losses["capt_loss"] = 0.0

    #     # Train policy
    #     num_updates = self.ppo_epoch * self.n_mini_batch
    #     for _ in range(self.ppo_epoch):
    #         data_generator = self.buffer.recurrent_policy_generator(
    #             act_advantages, comm_advantages)
    
    #         for sample in data_generator:
    #             if self.share_params:
    #                 loss = self._train_mappo(
    #                     0, sample, train_comm_head, train_lang)
                    
    #                 for key in loss:
    #                     losses[key] += loss[key] / num_updates
    #             else:
    #                 for a_i in range(len(self.agents)):
    #                     sample_i = (
    #                         *[batch[:, a_i] for batch in sample[:-1]],
    #                         [s[a_i] for s in sample[-1]])
    #                         # sample[-1][a_i])

    #                     loss = self._train_mappo(
    #                         a_i, 
    #                         sample_i, 
    #                         train_comm_head, 
    #                         train_lang)
                        
    #                     for key in loss:
    #                         losses[key] += loss[key] / (
    #                             num_updates * len(self.agents))

    #     return losses