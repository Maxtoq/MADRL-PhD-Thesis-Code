import torch
import numpy as np
import torch.nn as nn

from .utils import huber_loss
from .valuenorm import ValueNorm


class ACC_Trainer:

    def __init__(self, 
            args, agents, lang_learner, buffer, device=torch.device("cpu")):
        self.agents = agents
        self.lang_learner = lang_learner
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
        self.clip_batch_size = args.lang_clip_batch_size
        self.clip_epochs = args.lang_clip_epochs
        self.temp = args.lang_temp
        self.clip_weight = args.lang_clip_weight
        self.capt_weight = args.lang_capt_weight

        self.clip_loss = nn.CrossEntropyLoss()
        self.captioning_loss = nn.NLLLoss()

        self.value_normalizer = ValueNorm(1).to(device)

    def _compute_advantages(self):
         # Compute and normalize advantages
        advantages = self.buffer.returns[:-1] - self.value_normalizer.denormalize(
            self.buffer.value_preds[:-1])
        advantages_copy = advantages.copy()
        mean_advantages = np.nanmean(advantages_copy)
        std_advantages = np.nanstd(advantages_copy)
        advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)
        return advantages

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
        for d_o, e_t in zip(preds, targets):
            e_t = torch.argmax(e_t, dim=1).to(self.device)
            dec_loss += self.captioning_loss(d_o[:e_t.size(0)], e_t)
        return dec_loss

    # def _compute_language_losses(self, 
    #         agent, obs_batch, parsed_obs_batch, policy_input_batch, 
    #         rnn_states_batch, masks_batch):
    #     """
    #     :param agent: (ACC_Agent) Agent model used in the update.
    #     :param obs_batch: (torch.Tensor) Batch of observations, 
    #         dim=(ep_length * n_mini_batch * n_agents, obs_dim).
    #     :param parsed_obs_batch: (list(list(str))) Batch of sentences parsed 
    #         from observations, one for each observation in batch.
    #     """
    #     # Encode observations
    #     obs_context_batch = self.lang_learner.obs_encoder(obs_batch)
    #     # Encode sentences
    #     lang_context_batch = self.lang_learner.lang_encoder(parsed_obs_batch)
    #     lang_context_batch = lang_context_batch.squeeze()
    #     # CLIP loss
    #     clip_loss, mean_sim = self._compute_clip_loss(
    #         obs_context_batch, lang_context_batch)

    #     # Pass through acc
    #     comm_actions = agent.get_comm_actions(
    #         policy_input_batch, rnn_states_batch, masks_batch)
    #     # Decode
    #     encoded_targets = self.lang_learner.word_encoder.encode_batch(
    #         parsed_obs_batch)
    #     decoder_outputs, _ = self.lang_learner.decoder(
    #         comm_actions, encoded_targets)
    #     # Captioning loss
    #     capt_loss = self._compute_capt_loss(decoder_outputs, encoded_targets)
        
    #     return clip_loss, capt_loss, mean_sim

    def _update_policy(self, agent, sample, train_comm_head):
        policy_input_batch, critic_input_batch, rnn_states_batch, \
            critic_rnn_states_batch, env_actions_batch, comm_actions_batch, \
            old_env_action_log_probs_batch, old_comm_action_log_probs_batch, \
            value_preds_batch, returns_batch, masks_batch, advantages_batch \
                = sample

        policy_input_batch = torch.from_numpy(policy_input_batch).to(self.device)
        critic_input_batch = torch.from_numpy(critic_input_batch).to(self.device)
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
        # obs_batch = torch.from_numpy(obs_batch).to(self.device)

        # Agent forward pass
        values, env_action_log_probs, env_dist_entropy, comm_action_log_probs, \
            comm_dist_entropy = agent.evaluate_actions(
                policy_input_batch, critic_input_batch, rnn_states_batch, 
                critic_rnn_states_batch, env_actions_batch, comm_actions_batch, 
                masks_batch, train_comm_head)

        # Actor loss
        actor_loss = self._compute_policy_loss(
            env_action_log_probs, 
            old_env_action_log_probs_batch, 
            advantages_batch, 
            env_dist_entropy)

        log_losses = {"actor_loss": actor_loss.item()}

        # Communicator loss
        if train_comm_head:
            comm_loss = self._compute_policy_loss(
                comm_action_log_probs, 
                old_comm_action_log_probs_batch, 
                advantages_batch, 
                comm_dist_entropy)
            log_losses["comm_loss"] = comm_loss.item()
        else:
            comm_loss = torch.zeros_like(actor_loss)

        # Value loss
        value_loss = self._compute_value_loss(
            values, value_preds_batch, returns_batch)

        log_losses["value_loss"] = value_loss.item()

        loss = actor_loss + comm_loss + value_loss

        # # Language losses
        # if train_lang:
        #     clip_loss, capt_loss, mean_sim = self._compute_language_losses(
        #         agent, 
        #         obs_batch, 
        #         parsed_obs_batch, 
        #         policy_input_batch, 
        #         rnn_states_batch,
        #         masks_batch)
        #     loss += self.clip_weight * clip_loss + self.capt_weight * capt_loss

        #     log_losses["clip_loss"] = clip_loss.item()
        #     log_losses["capt_loss"] = capt_loss.item()
        #     log_losses["mean_sim"] = mean_sim

        
        # Update
        agent.act_comm_optim.zero_grad()
        agent.critic_optim.zero_grad()
        # self.lang_learner.optim.zero_grad()
        loss.backward()
        agent.act_comm_optim.step()
        agent.critic_optim.step()
        # self.lang_learner.optim.step()

        return log_losses

    def _update_language(self, sample):
        policy_input_batch, mask_batch, rnn_states_batch, obs_batch, \
            parsed_obs_batch = sample

        print(policy_input_batch.shape)
        if not self.share_params:
            print(obs_batch.shape)
            print(len(parsed_obs_batch), len(parsed_obs_batch[0]))
        exit()

    def train(self, warmup=False, train_comm_head=True, train_lang=True):
        """
        Train LGMARL.
        :param train_comm_head: (bool) whether to train the communicator head.

        :return losses: (dict) contains information regarding training 
            update (e.g. loss, grad norms, etc).
        """
        for a in self.agents:
            a.warmup_lr(warmup)
            
        advantages = self._compute_advantages()
        
        losses = {
            "value_loss": 0.0,
            "actor_loss": 0.0}
        if train_comm_head:
            losses["comm_loss"] = 0.0
        if train_lang:
            losses["clip_loss"] = 0.0
            losses["capt_loss"] = 0.0
            losses["mean_sim"] = 0.0

        # Train policy
        num_updates = self.ppo_epoch * self.n_mini_batch
        for _ in range(self.ppo_epoch):
            data_generator = self.buffer.recurrent_policy_generator(advantages)
    
            for sample in data_generator:
                if self.share_params:
                    loss = self._update_policy(
                        self.agents[0], sample, train_comm_head)
                    
                    for key in loss:
                        losses[key] += loss[key] / num_updates
                else:
                    for a_i in range(len(self.agents)):
                        sample_i = tuple(
                            [batch[:, a_i] for batch in sample])

                        loss = self._update_policy(
                            self.agents[a_i], sample_i, train_comm_head)
                        
                        for key in loss:
                            losses[key] += loss[key] / (
                                num_updates * len(self.agents))

        # Train language
        if train_lang:
            sample = self.buffer.sample_language()
            loss = self._update_language(sample)

            for key in loss:
                losses[key] += loss[key]
            # if self.share_params:
            #     loss = self._update_language(self.agents[0], sample)

            #     for key in loss:
            #         losses[key] += loss[key]
            # else:
            #     for a_i in range(len(self.agents)):
            #         # sample_i = (
            #         #     *[batch[:, a_i] for batch in sample[:-1]],
            #         #     sample[-1][a_i])

            #         loss = self._update_language(self.agents[a_i], sample)

            #         for key in loss:
            #             losses[key] += loss[key] / len(self.agents)
 
        return losses