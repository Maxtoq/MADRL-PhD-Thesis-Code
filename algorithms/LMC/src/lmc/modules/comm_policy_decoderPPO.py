import copy
import torch
import random
import itertools
import numpy as np

from torch import nn

from src.lmc.modules.networks import MLPNetwork, init
from src.lmc.utils import torch2numpy


# class Tester:

#     def __init__(self):
#         self.test_output = None

#     def test(self, module, input_tensor, name, rec_input=None):
#         if rec_input is not None:
#             _, test = module(rec_input, torch.ones_like(input_tensor))
#         else:
#             test = module(torch.ones_like(input_tensor))

#         if self.test_output is None:
#             self.test_output = test
#         else:
#             if not torch.all(torch.eq(self.test_output, test)):
#                 torch.set_printoptions(profile="full")
#                 print(name, "output CHANGED")
#             else:
#                 print(name, "output not changed")
    

class CommBuffer_MLP:
    
    def __init__(self, context_dim, max_sent_len, token_dim, gamma=0.99, 
            n_mini_batch=2):
        self.context_dim = context_dim
        self.max_sent_len = max_sent_len
        self.token_dim = token_dim
        assert gamma <= 1
        self.gamma = gamma
        self.n_mini_batch = n_mini_batch
        
        self.input_context = None
        self.generated_tokens = None
        self.token_log_probs = None
        self.value_preds = None
        self.returns = None
        self.masks = None
        
    def reset(self):
        self.input_context = None
        self.generated_tokens = None
        self.token_log_probs = None
        self.value_preds = None
        self.returns = None
        self.masks = None
        
    def store_gen(self, input_context, gen_tokens, token_log_probs, 
            value_preds, masks):
        """
        Store a batch of generated sequences.
        """
        self.input_context = input_context
        self.generated_tokens = gen_tokens
        self.token_log_probs = token_log_probs
        self.value_preds = value_preds
        self.masks = masks            
    
    def compute_returns(self, step_rewards):
        """
        Compute returns at each step of generations from final rewards.
        :param rewards (np.ndarray): Rewards given for each generated 
            sentences, dim=(seq_len, batch_size)
        """
        self.returns = step_rewards.copy()
        
        for s_i in reversed(range(self.returns.shape[0] - 1)):
            self.returns[s_i] = self.returns[s_i + 1] * self.gamma + \
                step_rewards[s_i]
        self.returns = self.returns[..., np.newaxis]
        
    def generator(self):
        """
        Randomise experiences and yields mini-batches to train.
        """
        # Each element of the batch is a data_chunk (one env step = generated 
        # sequence)
        batch_size = self.input_context.shape[0]
        assert batch_size % self.n_mini_batch == 0
        mini_batch_size = batch_size // self.n_mini_batch
        
        # Compute and normalise advantages
        advantages = self.returns[:-1] - self.value_preds[:-1]
        advantages = (advantages - np.mean(advantages)) / (
            np.std(advantages) + 1e-8)
        
        # Return mini_batches with permutated indexes
        rand_ids = np.random.permutation(batch_size)
        sample_ids = [
            rand_ids[i * mini_batch_size:(i + 1) * mini_batch_size] 
            for i in range(self.n_mini_batch)]
        for ids in sample_ids:
            yield self.input_context[ids], self.generated_tokens[:, ids], \
                  self.token_log_probs[:, ids], self.value_preds[:, ids], \
                  self.returns[:, ids], advantages[:, ids], self.masks[:, ids]


class TextActorCritic(nn.Module):
    
    def __init__(self, word_encoder, pretrained_decoder, context_dim, 
            max_sent_len, device, train_topk=1):
        super(TextActorCritic, self).__init__()
        self.word_encoder = word_encoder
        self.max_sent_len = max_sent_len
        self.device = device
        # TODO add topk param handle
        self.train_topk = train_topk
        # RNN encoder
        self.gru = copy.deepcopy(pretrained_decoder.gru)
        # Policy and value heads
        self.actor = copy.deepcopy(pretrained_decoder.out)
        self.critic = init(nn.Linear(context_dim, 1), gain=0.01)
            
    def gen_messages(self, context_batch):
        """
        :param context_batch (torch.Tensor): Batch of context vectors,
                dim=(1, batch_size, context_dim).
        """
        batch_size = context_batch.shape[1]
        # Set initial hidden states and token
        hidden = context_batch
        last_tokens = torch.tensor(
            np.array([[self.word_encoder.SOS_ENC]])).float().repeat(
                1, batch_size, 1).to(self.device)
        
        batch_tokens = []
        batch_log_probs = []
        batch_token_log_probs = []
        batch_value_preds = []
        batch_masks = [np.ones(batch_size)]
        last_topi = torch.zeros(batch_size)
        # batch_len_sentences = np.zeros(batch_size)
        sentences = [[] for b_i in range(batch_size)]
        for t_i in range(self.max_sent_len):
            # Encode with RNN
            _, hidden = self.gru(last_tokens, hidden)
            
            # Get token predictions from actor
            log_probs = self.actor(hidden)
            
            # Get values from critic
            value_preds = self.critic(hidden)
            
            # Sample next token
            _, topi = log_probs.topk(1)
            topi = topi.squeeze()
            tokens = self.word_encoder.token_encodings[topi.cpu()]
            
            # Make token_log_prob
            token_log_probs = log_probs.gather(-1, topi.reshape(1, -1, 1))
            
            # Make mask: 1 if last token is not EOS and last mask is not 0
            masks = (
                np.where(last_topi.cpu() == self.word_encoder.EOS_ID, 0.0, 1.0) * \
                np.where(batch_masks[-1] == 0.0, 0.0, 1.0))
            last_topi = topi
            
            # Stop early if all sentences are finished
            if sum(masks) == 0:
                break
            
            # Add decoded tokens to sentences
            for b_i in range(batch_size):
                if masks[b_i] and topi[b_i] != self.word_encoder.EOS_ID:
                    sentences[b_i].append(
                        self.word_encoder.index2token(topi[b_i]))
            
            batch_tokens.append(tokens)
            batch_log_probs.append(torch2numpy(log_probs))
            batch_token_log_probs.append(torch2numpy(token_log_probs))
            batch_value_preds.append(torch2numpy(value_preds))
            batch_masks.append(masks)
            
            last_tokens = torch.Tensor(tokens).unsqueeze(0).to(self.device)
            
        # Compute last value
        _, hidden = self.gru(last_tokens, hidden)
        value_preds = self.critic(hidden)
        batch_value_preds.append(torch2numpy(value_preds))
        
        tokens = np.stack(batch_tokens, dtype=np.float32)
        log_probs = np.concatenate(batch_log_probs)
        token_log_probs = np.concatenate(batch_token_log_probs)
        value_preds = np.concatenate(batch_value_preds)
        masks = np.stack(batch_masks)
        
        return tokens, token_log_probs, value_preds, masks, sentences, log_probs
    
    def evaluate_tokens(self, context_batch, token_batch):
        """
        Evaluate generated tokens with the current policy and value.
        :param context_batch (torch.Tensor): Batch of communication contexts
            (initial hidden state of gru), dim=(1, batch_size, context_dim)
        :param token_batch (torch.Tensor): Batch of generated tokens, 
            dim=(seq_len, batch_size, token_dim)
        
        :return token_log_probs (torch.Tensor): Log-probabilities of given 
            tokens, dim=(seq_len, batch_size, 1)
        :return entropy (torch.Tensor): Entropy of the output probabilities, 
            dim=(1)
        :return value_preds (torch.Tensor): Value predictions, dim=(seq_len, 
            batch_size, 1)
        """
        # Add SOS token
        sos_tensor = torch.Tensor(
            np.array([self.word_encoder.SOS_ENC])).repeat(
                1, context_batch.shape[1], 1).to(self.device)
        input_tokens = torch.cat((sos_tensor, token_batch)).to(self.device)

        outputs, _ = self.gru(input_tokens, context_batch)
        
        # Get log_probs and entropy
        log_probs = self.actor(outputs)
        token_log_probs = log_probs.gather(
            -1, token_batch.argmax(-1).unsqueeze(-1))
        entropy = -(log_probs * torch.exp(log_probs)).mean()

        # Get values
        value_preds = self.critic(outputs)

        return token_log_probs, entropy, value_preds

class CommPPO_MLP:
    """ 
    Communication module with a recurrent context encoder, 
    a policy that generates sentences and a value that estimates
    the quality of the current state (previous hidden state).
    It is trained using PPO, fine-tuning a pretrained policy.
    """
    def __init__(self, args, n_agents, lang_learner, device="cpu"):
        self.n_agents = n_agents
        self.n_envs = args.n_parallel_envs
        self.lr = args.comm_lr
        self.n_epochs = args.comm_n_epochs
        self.ppo_clip_param = args.comm_ppo_clip_param
        self.entropy_coef = args.comm_entropy_coef
        self.vloss_coef = args.comm_vloss_coef
        self.max_grad_norm = args.comm_max_grad_norm
        self.n_mini_batch = args.comm_n_mini_batch
        self.device = device
        self.warming_up = False
        
        self.lang_learner = lang_learner
        
        self.context_encoder = MLPNetwork(
            2 * args.context_dim, args.context_dim, norm_in=False)
        
        self.comm_policy = TextActorCritic(
            lang_learner.word_encoder, 
            lang_learner.decoder, 
            args.context_dim, 
            args.comm_max_sent_len,
            device,
            args.comm_train_topk)
        
        self.optim = torch.optim.Adam(
            list(self.comm_policy.parameters()) + \
            list(self.context_encoder.parameters()), 
            lr=self.lr, eps=1e-5)
        
        self.buffer = CommBuffer_MLP(
            args.context_dim,
            args.comm_max_sent_len, 
            self.lang_learner.word_encoder.enc_dim,
            args.comm_gamma,
            self.n_mini_batch)

    def warmup_lr(self, warmup):
        if warmup != self.warming_up:
            lr = self.lr * 0.01 if warmup else self.lr
            if warmup:
                print("WARMING UP", lr)
            else:
                print("STOP WARMING UP", lr)
            for param_group in self.optim.param_groups:
                param_group['lr'] = lr
            self.warming_up = warmup

    def prep_rollout(self, device=None):
        if device is None:
            device = self.device
        self.context_encoder.eval()
        self.context_encoder.to(device)
        self.comm_policy.eval()
        self.comm_policy.to(device)
        self.comm_policy.device = device

    def prep_training(self):
        self.context_encoder.train()
        self.context_encoder.to(self.device)
        self.comm_policy.train()
        self.comm_policy.to(self.device)
        self.comm_policy.device = self.device
        
    @torch.no_grad()
    def _get_pretrain_probs(self, context_batch, token_batch):
        """
        Get reference token log-probalities from pre-trained decoder.
        :param context_batch (np.ndarray): Batch of communication contexts
            (initial hidden state of gru), dim=(1, batch_size, context_dim)
        :param token_batch (np.ndarray): Batch of generated tokens, 
            dim=(seq_len, batch_size, token_dim)
        
        :return token_log_probs (torch.Tensor): Log-probabilities of given 
            tokens, dim=(seq_len, batch_size, 1)
        :return entropy (torch.Tensor): Entropy of the output probabilities, 
            dim=(1)
        :return value_preds (torch.Tensor): Value predictions, dim=(seq_len, 
            batch_size, 1)
        """
        token_batch = torch.from_numpy(token_batch).to(self.device)
        # Add SOS token
        sos_token = torch.Tensor(np.tile(
            self.lang_learner.word_encoder.SOS_ENC, 
            (1, context_batch.shape[1], 1))).to(self.device)
        input_tokens = torch.cat((sos_token, token_batch[:-1]))

        ref_log_probs, _ = self.lang_learner.decoder.forward_step(
            input_tokens, context_batch)
        
        return ref_log_probs
        
    @torch.no_grad()
    def get_messages(self, obs, lang_contexts):
        """
        Perform a communication step: encodes obs and previous messages and
        generates messages for this step.
        :param obs (np.ndarray): agents' observations for all parallel 
            environments, dim=(n_envs, n_agents, obs_dim)
            
        :return comm (list(list(str))): messages generated for each agent,
            for each parallel environment
        :return lang_context (np.ndarray): language context vectors to send
            to policy, dim=(n_envs, n_agents, context_dim)
        """
        # Encode inputs
        obs = torch.Tensor(obs).view(self.n_envs * self.n_agents, -1)
        obs_context = self.lang_learner.encode_observations(obs)

        # Repeat lang_contexts for each agent in envs
        lang_contexts = torch.from_numpy(lang_contexts.repeat(
            self.n_agents, 0).reshape(self.n_envs * self.n_agents, -1)).to(
                self.device)

        input_context = torch.cat((obs_context, lang_contexts), dim=-1)
        
        # Encode contexts
        comm_context = obs_context.unsqueeze(0) # NOCOMMENC self.context_encoder(input_context).unsqueeze(0)
        
        # Generate messages
        tokens, token_log_probs, value_preds, masks, messages, log_probs = \
            self.comm_policy.gen_messages(comm_context)
        
        # Compute KL-pretrain rewards
        # Get reference token_log_probs from pretrained decoder
        ref_log_probs = self._get_pretrain_probs(comm_context, tokens)
        # Compute KL divergence
        kl = (np.exp(log_probs) * (log_probs - torch2numpy(ref_log_probs))).sum(-1)
        
        # Store experiences in buffer
        self.buffer.store_gen(
            torch2numpy(input_context), 
            tokens, 
            token_log_probs, 
            value_preds, 
            masks)
        
        return messages, -kl

    # def _rand_filter_messages(self, messages):
    #     """
    #     Randomly filter out perfect messages.
    #     :param messages (list(list(list(str)))): Perfect messages, ordered by
    #         environment, by agent.

    #     :return filtered_broadcast (list(list(str))): Filtered message to 
    #         broadcast, one for each environment.
    #     """
    #     filtered_broadcast = []
    #     for env_messages in messages:
    #         env_broadcast = []
    #         for message in env_messages:
    #             if random.random() < 0.2:
    #                 env_broadcast.extend(message)
    #         filtered_broadcast.append(env_broadcast)
    #     return filtered_broadcast
    
    @torch.no_grad()
    def comm_step(self, obs, lang_contexts, perfect_messages=None):
        # Get messages
        messages, klpretrain_rewards = self.get_messages(obs, lang_contexts)
        
        # Arrange messages by env
        broadcasts = []
        messages_by_env = []
        for e_i in range(self.n_envs):
            env_broadcast = []
            for a_i in range(self.n_agents):
                env_broadcast.extend(messages[e_i * self.n_agents + a_i])
            broadcasts.append(env_broadcast)
            messages_by_env.append(messages[
                e_i * self.n_agents:e_i * self.n_agents + self.n_agents])

        new_lang_contexts = self.lang_learner.encode_sentences(
            broadcasts).cpu().numpy()

        # # TEST with perfect messages
        # broadcasts = self._rand_filter_messages(perfect_messages)
        # new_lang_contexts = self.lang_learner.encode_sentences(broadcasts).detach().cpu().numpy()
        
        # Return messages and lang_context
        return broadcasts, messages_by_env, new_lang_contexts, \
               klpretrain_rewards
    
    def store_rewards(self, message_rewards, token_rewards):
        """
        Send rewards for each sentences to the buffer to compute returns.
        :param message_rewards (np.ndarray): Rewards for each generated 
            sentence, dim=(n_agents * n_envs, )
        :param token_rewards (np.ndarray): Rewards for each generated token, 
            dim=(seq_len, n_agents * n_envs, 1)

        :return mean_message_return (float): Average return of evaluated 
            messages.
        """
        # print("message_rewards", message_rewards, message_rewards.shape)
        # print("token_rewards", token_rewards, token_rewards.shape)
        token_rewards *= self.buffer.masks[1:]
        # print("token_rewards", token_rewards, token_rewards.shape)

        len_sentences = np.sum(self.buffer.masks, axis=0, dtype=int)
        step_rewards = np.zeros_like(self.buffer.masks)
        # print("step_rewards", step_rewards, step_rewards.shape)
        # Set final reward to final token of each sentence
        step_rewards[len_sentences - 1, list(range(message_rewards.shape[0]))] = \
            message_rewards
        # print("step_rewards", step_rewards, step_rewards.shape)
        # Add klpretrain penalty
        step_rewards[1:] += token_rewards
        # print("step_rewards", step_rewards, step_rewards.shape)
        
        # Clean masked rewards
        step_rewards *= self.buffer.masks
        # print("buffer.masks", self.buffer.masks)
        # print("step_rewards", step_rewards, step_rewards.shape)

        self.buffer.compute_returns(step_rewards)

        # print("kl_reward", token_rewards.mean())
        # print("token_reward", step_rewards.sum() / self.buffer.masks.sum())
        # print("returns", self.buffer.returns, self.buffer.returns.shape)
        # exit()

        # mean_step_rewards = step_rewards.sum() / step_rewards.shape[1]

        rewards = {
            "kl_reward": token_rewards.mean(),
            "env_reward": message_rewards.mean(),
            "token_reward": step_rewards.sum() / self.buffer.masks.sum()
        }

        # return step_rewards.sum() / step_rewards.shape[1]
        return rewards
        
    def ppo_update(self, batch):
        input_context, generated_tokens, token_log_probs, value_preds, returns, \
            advantages, masks = batch
        
        input_context = torch.from_numpy(input_context).to(self.device)
        generated_tokens = torch.from_numpy(generated_tokens).to(self.device)
        token_log_probs = torch.from_numpy(token_log_probs).to(self.device)
        advantages = torch.from_numpy(advantages).to(self.device)
        returns = torch.from_numpy(returns).to(self.device)
        
        # Evaluate generated tokens
        # NOCOMMENC comm_context = self.context_encoder(input_context).unsqueeze(0)
        comm_context = input_context[:, :16].unsqueeze(0).contiguous()
        new_token_log_probs, entropy, new_value_preds = \
            self.comm_policy.evaluate_tokens(comm_context, generated_tokens)
            
        # Policy Loss
        ratio = (new_token_log_probs - token_log_probs).exp()
        pol_loss1 = -advantages * ratio
        pol_loss2 = -advantages * torch.clamp(
            ratio, 1 - self.ppo_clip_param, 1 + self.ppo_clip_param)
        pol_loss = torch.max(pol_loss1, pol_loss2).mean()
        
        # Entropy loss
        entropy_loss = entropy.mean()
        
        # Value loss
        val_loss = ((new_value_preds - returns) ** 2).mean()
        
        # Final PPO loss
        loss = pol_loss - self.entropy_coef * entropy_loss + \
                self.vloss_coef * val_loss
        
        # Update
        self.optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            self.comm_policy.parameters(), self.max_grad_norm)
        nn.utils.clip_grad_norm_(
            self.context_encoder.parameters(), self.max_grad_norm)
        self.optim.step()

        return pol_loss, entropy_loss, val_loss
    
    def train(self, warmup=False):
        self.warmup_lr(warmup)
        self.prep_training()
        losses = {
            "comm_policy_loss": 0,
            "comm_entropy_loss": 0,
            "comm_value_loss": 0}
        for e_i in range(self.n_epochs):
            data_generator = self.buffer.generator()
            
            for train_batch in data_generator:
                pol_loss, entropy_loss, val_loss = self.ppo_update(train_batch)
                
                losses["comm_policy_loss"] += pol_loss.item()
                losses["comm_entropy_loss"] += entropy_loss.item()
                losses["comm_value_loss"] += val_loss.item()
        
        for k in losses.keys():
            losses[k] /= (self.n_epochs * self.n_mini_batch)
        
        self.prep_rollout()

        return losses

    def get_save_dict(self):
        save_dict = {
            "context_encoder": self.context_encoder.state_dict(),
            "comm_policy": self.comm_policy.state_dict(),
            "comm_optim": self.optim.state_dict()}
        return save_dict

    def load_params(self, save_dict):
        self.lang_learner.load_params(save_dict)
        if "context_encoder" in save_dict:
            self.context_encoder.load_state_dict(save_dict["context_encoder"])
            self.comm_policy.load_state_dict(save_dict["comm_policy"])
            self.optim.load_state_dict(save_dict["comm_optim"])
        else: # Starting fine-tuning from pretrained language learner
            self.comm_policy.gru.load_state_dict(
                self.lang_learner.decoder.gru.state_dict())
            self.comm_policy.actor.load_state_dict(
                self.lang_learner.decoder.out.state_dict())
