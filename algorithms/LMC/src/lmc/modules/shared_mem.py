import numpy as np
import torch
import torch.nn.functional as F

from torch import nn

from .lm import init_rnn_params
from .networks import MLPNetwork
from src.lmc.utils import torch2numpy


class SharedMemoryBuffer():

    def __init__(self, args, state_dim):
        self.n_parallel_envs = args.n_parallel_envs
        self.max_size = args.shared_mem_max_buffer_size
        self.max_ep_length = args.episode_length
        self.batch_size = args.shared_mem_batch_size

        self.message_enc_buffer = np.zeros(
            (self.max_size, self.max_ep_length, args.context_dim))
        self.state_buffer = np.zeros(
            (self.max_size, self.max_ep_length, state_dim))
        self.ep_lens = np.zeros(self.max_size, dtype=np.int32)
        self.finished_eps = np.zeros(self.max_size, dtype=np.int32)

        self.current_ids = np.arange(self.n_parallel_envs, dtype=np.int32)

        self.current_size = self.n_parallel_envs

    def _reset_buffer_entries(self, ids):
        self.message_enc_buffer[ids]

    def end_episode(self, dones):
        # Find ids of finished episodes
        finished_ep_ids = self.current_ids[dones]
        self.finished_eps[finished_ep_ids] += 1

        # Roll if needed
        n_changes = len(finished_ep_ids)
        if n_changes + self.current_size > self.max_size:
            n_shift = n_changes + self.current_size - self.max_size
            # Roll buffers and init rolled entries to zeros
            self.message_enc_buffer = np.roll(
                self.message_enc_buffer, -n_shift, axis=0)
            self.message_enc_buffer[-n_shift:] *= 0
            self.state_buffer = np.roll(self.state_buffer, -n_shift, axis=0)
            self.state_buffer[-n_shift:] *= 0
            self.ep_lens = np.roll(self.ep_lens, -n_shift, axis=0)
            self.ep_lens[-n_shift:] *= 0
            self.finished_eps = np.roll(self.finished_eps, -n_shift, axis=0)
            self.finished_eps[-n_shift:] *= 0
            # Update ids accordingly
            self.current_ids -= n_shift

        # Set new ids
        for i in range(self.n_parallel_envs):
            if dones[i]:
                self.current_ids[i] = self.current_ids.max() + 1
        
        self.current_size = min(self.current_size + n_changes, self.max_size)

    def store(self, message_encodings, states):
        """
        Store step.
        :param message_encodings: (np.ndarray) Encoded messages, 
            dim=(n_parallel_envs, context_dim).
        :param states: (np.ndarray) Actual states, dim=(n_parallel_envs, 
            state_dim).
        """
        self.message_enc_buffer[
            self.current_ids, 
            self.ep_lens[self.current_ids]] = message_encodings
        self.state_buffer[
            self.current_ids, 
            self.ep_lens[self.current_ids]] = states

        self.ep_lens[self.current_ids] += 1

    def sample(self):
        """
        Sample a batch of training data.
        :return message_encodings: (np.ndarray) Encoded messages, 
            dim=(batch_size, max_ep_len, context_dim).
        :return states: (np.ndarray) Actual states, dim=(batch_size, 
            max_ep_len, state_dim).
        :return ep_lens: (np.ndarray) Actual length of each sampled episode,
            dim=(batch_size,).
        """
        n_finished = self.finished_eps.sum()
        batch_size = min(self.batch_size, n_finished)

        sample_ids = np.random.choice(
            n_finished, batch_size, replace=False)
            
        return self.message_enc_buffer[sample_ids], \
               self.state_buffer[sample_ids], \
               self.ep_lens[sample_ids]


class SharedMemory():

    def __init__(self, args, n_agents, state_dim, device="cpu"):
        self.n_agents = n_agents
        self.hidden_dim = args.shared_mem_hidden_dim
        self.n_rec_layers = args.shared_mem_n_rec_layers
        self.n_parallel_envs = args.n_parallel_envs
        self.state_dim = state_dim
        self.device = device

        # # Language Encoder
        # self.lang_encoder = lang_encoder

        # GRU layer
        self.gru = nn.GRU(
            args.context_dim, self.hidden_dim, self.n_rec_layers)
        init_rnn_params(self.gru)
        self.norm = nn.LayerNorm(self.hidden_dim)

        # Output prediction layer
        self.out = MLPNetwork(
            self.hidden_dim, state_dim, self.hidden_dim, norm_in=None)

        # Optimizer and loss
        self.optim = torch.optim.Adam(
            # list(self.lang_encoder.parameters()) \
            list(self.gru.parameters()) \
            + list(self.out.parameters()), 
            lr=args.shared_mem_lr)

        # Buffer
        self.buffer = SharedMemoryBuffer(args, state_dim)

        self.memory_context = torch.zeros(
            self.n_rec_layers, 
            self.n_parallel_envs, 
            self.hidden_dim).to(self.device)

    def prep_rollout(self, device=None):
        if device is None:
            device = self.device
        else:
            self.device = device
        # self.lang_encoder.eval()
        # self.lang_encoder.to(device)
        # self.lang_encoder.device = device
        self.gru.eval()
        self.gru.to(device)
        self.out.eval()
        self.out.to(device)
        self.norm.eval()
        self.norm.to(device)
        self.memory_context = self.memory_context.to(device)

    def prep_training(self, device=None):
        if device is None:
            device = self.device
        else:
            self.device = device
        # self.lang_encoder.train()
        # self.lang_encoder.to(device)
        # self.lang_encoder.device = device
        self.gru.train()
        self.gru.to(device)
        self.out.train()
        self.out.to(device)
        self.norm.train()
        self.norm.to(device)
        self.memory_context = self.memory_context.to(device)

    def reset_context(self, env_dones=None, n_envs=None):
        if env_dones is None:
            if n_envs is None:
                n_envs = self.n_parallel_envs
            self.memory_context = torch.zeros(
                self.n_rec_layers, n_envs, self.hidden_dim).to(self.device)
        else:
            self.memory_context = self.memory_context * torch.Tensor(
                (1 - env_dones).astype(np.float32).reshape(
                    (1, self.n_parallel_envs, 1))).to(self.device)
        
            # Change place to store in buffer for finished episode
            self.buffer.end_episode(env_dones)

    def store(self, message_encodings, states):
        self.buffer.store(message_encodings, states)

    def _predict_states(self, message_encodings, hidden_states):
        """
        Predict states from messages.
        :param message_encodings: (torch.Tensor) Encoded messages, 
            dim=(seq_len, n_parallel_envs, context_dim).
        :param hidden_states: (torch.Tensor) Hidden states of the GRU,
            dim=(n_rec_layers, n_parallel_envs, hidden_dim).

        :return pred_states: (torch.Tensor) Actual states, dim=(seq_len, 
            n_parallel_envs, state_dim).
        :return hidden_states: (torch.Tensor) New hidden states of the GRU,
            dim=(n_rec_layers, n_parallel_envs, hidden_dim).
        """
        x, hidden_states = self.gru(
            message_encodings.to(self.device), hidden_states.to(self.device))

        if type(x) is torch.nn.utils.rnn.PackedSequence:
            x = torch.nn.utils.rnn.unpack_sequence(x)
            x = nn.utils.rnn.pad_sequence(x)

        x = self.norm(x)

        pred_states = self.out(x)
        return pred_states, hidden_states

    @torch.no_grad()
    def get_prediction_error(self, 
            message_encodings, broadcast_encodings, states):
        """
        Return the prediction error of the model given communicated messages
        and actual states.
        :param message_encodings: (np.ndarray) Encoded agent messages, 
            dim=(n_parallel_envs * n_agents, context_dim).
        :param broadcast_encodings: (np.ndarray) Encoded broadcasted messages, 
            dim=(n_parallel_envs, context_dim).
        :param states: (np.ndarray) Actual states, dim=(n_parallel_envs, 
            state_dim).

        :return local_errors: (np.ndarray) Prediction errors given agent
            messages, dim=(n_parallel_envs * n_agents,).
        :return common_errors: (np.ndarray) Prediction error given broadcasted
            messages, dim=(n_parallel_envs,).
        """
        # Compute error given each agent's message
        pred_states, _ = self._predict_states(
            torch.Tensor(message_encodings).unsqueeze(0).to(self.device),
            self.memory_context.repeat(1, 1, self.n_agents).reshape(
                1, self.n_parallel_envs * self.n_agents, -1))
        local_errors = np.linalg.norm(
            torch2numpy(pred_states.squeeze(0)) \
                - states.repeat(self.n_agents, axis=0), 
            2, axis=-1)

        # Forward pass with broadcasted messages to get memory
        common_pred_states, self.memory_context = self._predict_states(
            torch.Tensor(broadcast_encodings).unsqueeze(0).to(self.device),
            self.memory_context)
        common_errors = np.linalg.norm(
            torch2numpy(common_pred_states.squeeze(0)) - states, 2, axis=-1)
        
        return local_errors, common_errors

    def train(self):
        """
        Train the model.
        :return loss: (float) Training loss.
        """
        message_encodings, states, ep_lens = self.buffer.sample()
        
        # Sort by episode length decreasing
        sorted_ids = np.argsort(ep_lens)[::-1]
        sorted_mess_enc = [
            torch.Tensor(message_encodings[s_i, :ep_lens[s_i]])
            for s_i in sorted_ids]

        # Pad and pack sequences
        padded = nn.utils.rnn.pad_sequence(sorted_mess_enc)
        packed = nn.utils.rnn.pack_padded_sequence(
            padded, ep_lens[sorted_ids]).to(self.device)

        hidden = torch.zeros(
            (self.n_rec_layers, 
             len(sorted_mess_enc), 
             self.hidden_dim)).to(self.device)

        # Predict states
        pred_states, _ = self._predict_states(packed, hidden)

        # Compute MSE Loss, without counting steps that did not happen
        sorted_states = torch.Tensor(
            states[sorted_ids]).transpose(0, 1).to(self.device)
        if pred_states.shape[0] < sorted_states.shape[0]:
            sorted_states = sorted_states[:pred_states.shape[0]]
        error = F.mse_loss(pred_states, sorted_states, reduction="sum")
        loss = error / ep_lens.sum()

        # Backward prop
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        return loss.item()

    def get_save_dict(self):
        save_dict = {
            "shared_mem_gru": self.gru.state_dict(),
            "shared_mem_out": self.out.state_dict()}
        return save_dict

    def load_params(self, save_dict):
        self.gru.load_state_dict(save_dict["shared_mem_gru"])
        self.out.load_state_dict(save_dict["shared_mem_out"])
