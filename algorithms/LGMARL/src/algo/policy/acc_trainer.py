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
        self.temp = args.lang_temp
        self.clip_weight = args.lang_clip_weight
        self.capt_weight = args.lang_capt_weight

        self.clip_loss = nn.CrossEntropyLoss()
        self.captioning_loss = nn.NLLLoss()

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
        norm_lang_context = lang_contexts / lang_contexts.norm(
            dim=1, keepdim=True)
        sim = norm_obs_context @ norm_lang_context.t() * self.temp
        mean_sim = sim.diag().mean()

        # Compute CLIP loss
        labels = torch.arange(len(obs_batch)).to(self.device)
        loss_o = self.clip_loss(sim, labels)
        loss_l = self.clip_loss(sim.t(), labels)
        clip_loss = (loss_o + loss_l) / 2

        return clip_loss, mean_sim.item()

    def _compute_language_losses(self, agent, obs_batch, parsed_obs_batch, policy_input_batch):
        """
        :param agent: (ACC_Agent) Agent model used in the update.
        :param obs_batch: (torch.Tensor) Batch of observations, 
            dim=(ep_length * n_mini_batch * n_agents, obs_dim).
        :param parsed_obs_batch: (list(list(str))) Batch of sentences parsed 
            from observations, one for each observation in batch.
        """
        # Encode observations
        # print(obs_batch, obs_batch.shape)
        # print(len(parsed_obs_batch))
        for i in range(len(parsed_obs_batch)):
            print(i)
            print(obs_batch[i])
            print(parsed_obs_batch[i])
        # Encode observations
        obs_context_batch = self.lang_learner.obs_encoder(obs_batch)

        # Encode sentences
        lang_context_batch = self.lang_learner.lang_encoder(parsed_obs_batch)
        lang_context_batch = lang_context_batch.squeeze()

        # CLIP loss
        clip_loss, mean_sim = self._compute_clip_loss(obs_context_batch)

        print(obs_context, obs_context_batch.shape)
        print(lang_context, lang_context_batch.shape)

        # Make input to policy
        print(policy_input_batch, policy_input_batch.shape)
        exit()

    def _update(self, agent, sample, train_comm_head, train_lang):
        policy_input_batch, critic_input_batch, rnn_states_batch, critic_rnn_states_batch, \
            env_actions_batch, comm_actions_batch, old_env_action_log_probs_batch, \
            old_comm_action_log_probs_batch, value_preds_batch, returns_batch, \
            masks_batch, advantages_batch, obs_batch, parsed_obs_batch = sample

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
        obs_batch = torch.from_numpy(obs_batch).to(self.device)

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

        # Communicator loss
        if train_comm_head:
            comm_loss = self._compute_policy_loss(
                comm_action_log_probs, 
                old_comm_action_log_probs_batch, 
                advantages_batch, 
                comm_dist_entropy)
        else:
            comm_loss = torch.zeros_like(actor_loss)

        # Value loss
        value_loss = self._compute_value_loss(
            values, value_preds_batch, returns_batch)

        # Language losses
        if train_lang:
            clip_loss, dec_loss = self._compute_language_losses(
                agent, obs_batch, parsed_obs_batch, policy_input_batch)
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
                        sample_i = (
                            *[batch[:, a_i] for batch in sample[:-1]],
                            [step_sentences[a_i] 
                                for step_sentences in sample[-1]])

                        self._update(
                            self.agents[a_i], sample_i, train_comm_head, train_lang)
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