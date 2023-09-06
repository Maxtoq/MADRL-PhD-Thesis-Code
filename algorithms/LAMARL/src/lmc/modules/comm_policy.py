import copy
import torch
import random
import itertools
import numpy as np

from torch import nn

from src.lmc.modules.networks import MLPNetwork


def torch2numpy(x):
    return x.detach().cpu().numpy()
    

class CommBuffer_MLP:
    
    def __init__(self, context_dim, hidden_dim, max_sent_len, token_dim, gamma=0.99, n_mini_batch=2):
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim
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
        
    def store_gen(self, input_context, gen_tokens, token_log_probs, value_preds, masks):
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
        :param rewards (np.ndarray): Rewards given for each generated sentences, dim=(seq_len, batch_size)
        """
        self.returns = step_rewards.copy()
        
        for s_i in reversed(range(self.returns.shape[0] - 1)):
            self.returns[s_i] = self.returns[s_i + 1] * self.gamma + step_rewards[s_i]
        self.returns = self.returns[..., np.newaxis]
        
    def recurrent_generator(self):
        """
        Randomise experiences and yields mini-batches to train.
        """
        # Each element of the batch is a data_chunk (one env step = generated sequence)
        batch_size = self.input_context.shape[0]
        assert batch_size % self.n_mini_batch == 0
        mini_batch_size = batch_size // self.n_mini_batch
        
        # Compute and normalise advantages
        advantages = self.returns[:-1] - self.value_preds[:-1]
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
        
        # Return mini_batches with permutated indexes
        rand_ids = np.random.permutation(batch_size)
        sample_ids = [rand_ids[i * mini_batch_size:(i + 1) * mini_batch_size] for i in range(self.n_mini_batch)]
        for ids in sample_ids:
            yield self.input_context[ids], self.generated_tokens[:, ids], self.token_log_probs[:, ids], \
                  self.value_preds[:, ids], self.returns[:, ids], advantages[:, ids], self.masks[:, ids]


class TextActorCritic(nn.Module):
    
    def __init__(self, word_encoder, pretrained_decoder, context_dim, max_sent_len):
        super(TextActorCritic, self).__init__()
        self.word_encoder = word_encoder
        self.max_sent_len = max_sent_len
        # RNN encoder
        self.gru = copy.deepcopy(pretrained_decoder.gru)
        # Policy and value heads
        self.actor = copy.deepcopy(pretrained_decoder.out)
        self.critic = nn.Linear(context_dim, 1)
            
    def gen_messages(self, context_batch):
        """
        :param context_batch (torch.Tensor): Batch of context vectors,
                dim=(1, batch_size, context_dim).
        """
        batch_size = context_batch.shape[1]
        # Set initial hidden states and token
        hidden = context_batch
        last_tokens = torch.tensor(
            np.array([[self.word_encoder.SOS_ENC]])).float().repeat(1, batch_size, 1)
        
        batch_tokens = []
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
            tokens = self.word_encoder.token_encodings[topi]
            
            # Make token_log_prob
            token_log_probs = log_probs.gather(-1, topi.reshape(1, -1, 1))
            
            # Make mask: 1 if last token is not EOS and last mask is not 0
            masks = (np.where(last_topi == self.word_encoder.EOS_ID, 0.0, 1.0) * \
                     np.where(batch_masks[-1] == 0.0, 0.0, 1.0))
            last_topi = topi
            
            # Stop early if all sentences are finished
            if sum(masks) == 0:
                break
            
            # Add decoded tokens to sentences
            for b_i in range(batch_size):
                if masks[b_i]:
                    sentences[b_i].append(self.word_encoder.index2token(topi[b_i]))
            
            batch_tokens.append(tokens)
            batch_token_log_probs.append(torch2numpy(token_log_probs))
            batch_value_preds.append(torch2numpy(value_preds))
            batch_masks.append(masks)
            
            last_tokens = torch.Tensor(tokens).unsqueeze(0)
            
        # Compute last value
        _, hidden = self.gru(last_tokens, hidden)
        value_preds = self.critic(hidden)
        batch_value_preds.append(torch2numpy(value_preds))
        
        tokens = np.stack(batch_tokens, dtype=np.float32)
        token_log_probs = np.concatenate(batch_token_log_probs)
        value_preds = np.concatenate(batch_value_preds)
        masks = np.stack(batch_masks)
        
        return tokens, token_log_probs, value_preds, masks, sentences
    
    def evaluate_tokens(self, context_batch, token_batch):
        """
        Evaluate generated tokens with the current policy and value.
        :param context_batch (torch.Tensor): Batch of communication contexts (initial hidden state of gru), 
            dim=(1, batch_size, context_dim)
        :param token_batch (torch.Tensor): Batch of generated tokens, dim=(seq_len, batch_size, token_dim)
        
        :return token_log_probs (torch.Tensor): Log-probabilities of given tokens, dim=(seq_len, batch_size, 1)
        :return entropy (torch.Tensor): Entropy of the output probabilities, dim=(1)
        :return value_preds (torch.Tensor): Value predictions, dim=(seq_len, batch_size, 1)
        """
        # Add SOS token
        sos_tensor = torch.Tensor(np.array([self.word_encoder.SOS_ENC])).repeat(1, context_batch.shape[1], 1)
        input_tokens = torch.cat((sos_tensor, token_batch))
        
        outputs, _ = self.gru(input_tokens, context_batch)
        
        # Get log_probs and entropy
        log_probs = self.actor(outputs)
        token_log_probs = log_probs.gather(-1, token_batch.argmax(-1).unsqueeze(-1))
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
    def __init__(self, n_agents, context_dim, hidden_dim, lang_learner, 
                 lr=0.0005, ep_len=10, max_sent_len=12, n_envs=1, gamma=0.99,
                 n_epochs=16, clip_coef=0.2, entropy_coef=0.01, vloss_coef=0.5, 
                 klpretrain_coef=0.01, max_grad_norm=10.0, n_mini_batch=2):
        self.n_agents = n_agents
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim
        self.n_envs = n_envs
        self.n_epochs = n_epochs
        self.clip_coef = clip_coef
        self.entropy_coef = entropy_coef
        self.vloss_coef = vloss_coef
        self.klpretrain_coef = klpretrain_coef
        self.max_grad_norm = max_grad_norm
        self.n_mini_batch = n_mini_batch
        
        self.lang_learner = lang_learner
        
        self.context_encoder = MLPNetwork(2 * context_dim, context_dim, norm_in=False)
        
        self.comm_policy = TextActorCritic(
            lang_learner.word_encoder, 
            lang_learner.decoder, 
            context_dim, 
            max_sent_len)
        
        self.optim = torch.optim.Adam(
            list(self.comm_policy.parameters()) + list(self.context_encoder.parameters()), 
            lr=lr, eps=1e-5)
        
        self.buffer = CommBuffer_MLP(
            context_dim, 
            hidden_dim, 
            max_sent_len, 
            self.lang_learner.word_encoder.enc_dim,
            gamma,
            n_mini_batch)
        
        self.last_comm = None
        
    def reset_episode(self):
        self.last_comm = None
    
    def save(self, path):
        pass
        
    @torch.no_grad()
    def get_messages(self, obs):
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
        obs_context = []
        obs = torch.Tensor(obs).view(self.n_envs * self.n_agents, -1)
        obs_context = self.lang_learner.encode_observations(obs)
        
        if self.last_comm is not None:
            sentences = list(itertools.chain.from_iterable(self.last_comm))
            lang_context = self.lang_learner.encode_sentences(sentences)
        else:
            lang_context = torch.zeros_like(obs_context)
            
        input_context = torch.cat((obs_context, lang_context), dim=-1)
        # Flatten rollout and agent dimensions
        #input_context = input_context.view(self.n_envs * self.n_agents, 2 * self.context_dim)
        
        # Encode contexts
        comm_context = self.context_encoder(input_context).unsqueeze(0)
        
        # Generate messages
        tokens, token_log_probs, value_preds, masks, sentences = self.comm_policy.gen_messages(comm_context)
        
        # Compute KL-pretrain rewards
        # Get reference token_log_probs from pretrained decoder
        ref_token_log_probs = self._get_pretrain_probs(comm_context, tokens)
        # Compute KL divergence
        klpretrain_rewards = -(token_log_probs - torch2numpy(ref_token_log_probs))
        
        # Store experiences in buffer
        self.buffer.store_gen(torch2numpy(input_context), tokens, token_log_probs, value_preds, masks)
        
        # Arrange sentences by env
        self.last_comm = [sentences[e_i * self.n_agents:(e_i + 1) * self.n_agents] for e_i in range(self.n_envs)]
        
        return self.last_comm, lang_context, klpretrain_rewards
        
    @torch.no_grad()
    def _get_pretrain_probs(self, context_batch, token_batch):
        """
        Get reference token log-probalities from pre-trained decoder.
        :param context_batch (np.ndarray): Batch of communication contexts (initial hidden state of gru), 
            dim=(1, batch_size, context_dim)
        :param token_batch (np.ndarray): Batch of generated tokens, dim=(seq_len, batch_size, token_dim)
        
        :return token_log_probs (torch.Tensor): Log-probabilities of given tokens, dim=(seq_len, batch_size, 1)
        :return entropy (torch.Tensor): Entropy of the output probabilities, dim=(1)
        :return value_preds (torch.Tensor): Value predictions, dim=(seq_len, batch_size, 1)
        """
        token_batch = torch.from_numpy(token_batch)
        # Add SOS token
        sos_token = np.tile(self.lang_learner.word_encoder.SOS_ENC, (1, context_batch.shape[1], 1))
        input_tokens = torch.Tensor(np.concatenate((sos_token, token_batch)))
        
        ref_log_probs, _ = self.lang_learner.decoder.forward_step(input_tokens, context_batch)
        ref_token_log_probs = ref_log_probs.gather(-1, token_batch.argmax(-1).unsqueeze(-1))
        return ref_token_log_probs
    
    def _store_rewards(self, message_rewards, klpretrain_rewards):
        """
        Send rewards for each sentences to the buffer to compute returns.
        :param message_rewards (np.ndarray): Rewards for each generated sentence, dim=(batch_size, )
        :param klpretrain_rewards (np.ndarray): Penalties for diverging from pre-trained decoder, 
            dim=(seq_len, batch_size, 1)
        """
        len_sentences = np.sum(self.buffer.masks, axis=0, dtype=int)
        step_rewards = np.zeros_like(self.buffer.masks)
        # Set final reward to final token of each sentence
        step_rewards[len_sentences - 1, list(range(message_rewards.shape[0]))] = message_rewards
        # Add klpretrain penalty
        step_rewards[1:] += self.klpretrain_coef * klpretrain_rewards.squeeze(-1)
        
        # Clean masked rewards
        step_rewards = step_rewards * self.buffer.masks
        
        self.buffer.compute_returns(step_rewards)
        
    def ppo_update(self, batch):
        input_context, generated_tokens, token_log_probs, value_preds, returns, advantages, masks = batch
        
        input_context = torch.from_numpy(input_context)
        generated_tokens = torch.from_numpy(generated_tokens)
        token_log_probs = torch.from_numpy(token_log_probs)
        advantages = torch.from_numpy(advantages)
        returns = torch.from_numpy(returns)
        
        # Evaluate generated tokens
        comm_context = self.context_encoder(input_context).unsqueeze(0)
        new_token_log_probs, entropy, new_value_preds = \
            self.comm_policy.evaluate_tokens(comm_context, generated_tokens)
            
        # Policy Loss
        ratio = (new_token_log_probs - token_log_probs).exp()
        pol_loss1 = -advantages * ratio
        pol_loss2 = -advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
        pol_loss = torch.max(pol_loss1, pol_loss2).mean()
        
        # Entropy loss
        entropy_loss = entropy.mean()
        
        # Value loss
        val_loss = ((new_value_preds - returns) ** 2).mean()
        
        # Final loss
        loss = pol_loss - self.entropy_coef * entropy_loss + \
                self.vloss_coef * val_loss
        
        # Update
        self.optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.comm_policy.parameters(), self.max_grad_norm)
        nn.utils.clip_grad_norm_(self.context_encoder.parameters(), self.max_grad_norm)
        self.optim.step()
        
        return pol_loss, entropy_loss, val_loss
    
    def train(self):
        losses = {
            "policy_loss": 0,
            "entropy_loss": 0,
            "value_loss": 0}
        for e_i in range(self.n_epochs):
            data_generator = self.buffer.recurrent_generator()
            
            for train_batch in data_generator:
                pol_loss, entropy_loss, val_loss = self.ppo_update(train_batch)
                
                losses["policy_loss"] += pol_loss.item()
                losses["entropy_loss"] += entropy_loss.item()
                losses["value_loss"] += val_loss.item()
        
        for k in losses.keys():
            losses[k] /= (self.n_epochs * self.n_mini_batch)
            
        return losses
    
    def comm_step(self, obs):
        # Get messages
        messages, lang_context, klpretrain_rewards = self.get_messages(obs)
        
        # Get rewards
        # TODO: add real rewards
        message_rewards = np.zeros(self.n_envs * self.n_agents)
        rewards = {
            "comm": message_rewards.mean(),
            "klpretrain": klpretrain_rewards.mean()}
        
        # Store rewards
        self._store_rewards(message_rewards, klpretrain_rewards)
        
        # Train
        losses = self.train()
        
        # Return messages and lang_context
        return messages, lang_context, rewards, losses


class PerfectComm:

    def __init__(self, lang_learner, prob_send_message=0.2):
        self.lang_learner = lang_learner
        self.prob_send_message = prob_send_message

    def _rand_filter_messages(self, messages):
        """
        Randomly filter out perfect messages.
        :param messages (list(list(list(str)))): Perfect messages, ordered by
            environment, by agent.

        :return filtered_broadcast (list(list(str))): Filtered message to 
            broadcast, one for each environment.
        """
        filtered_broadcast = []
        for env_messages in messages:
            env_broadcast = []
            for message in env_messages:
                if random.random() < self.prob_send_message:
                    env_broadcast.extend(message)
            filtered_broadcast.append(env_broadcast)
        return filtered_broadcast

    def comm_step(self, obs, perfect_messages):
        """
        Perform a communication step.
        :param obs (np.ndarray): Observations.
        :param perfect messages (list(list(list(str)))): Perfect messages,
            ordered by environment, by agent.

        :return broadcasts (list(list(str))): Broadcasted messages for each
            environment.
        :return next_contents (np.ndarray): Language contexts for next step,
            dim=(n_envs, context_dim).
        """
        # Determines the content of the broadcasted message
        broadcasts = self._rand_filter_messages(perfect_messages)
        
        # Compute next context
        next_contexts = self.lang_learner.encode_sentences(broadcasts)

        return broadcasts, next_contexts.detach().cpu().numpy()
