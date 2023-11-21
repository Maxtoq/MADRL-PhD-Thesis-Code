import numpy as np
import torch

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

        self.current_ids = np.arange(self.n_parallel_envs, dtype=np.int32)

        self.current_size = self.n_parallel_envs

    def _reset_buffer_entries(self, ids):
        self.message_enc_buffer[ids]

    def end_episode(self, dones):
        # Find ids of finished episodes
        finished_ep_ids = self.current_ids[dones]

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
            # Update ids accordingly
            self.current_ids -= n_shift

        # Set new ids
        for i in range(self.n_parallel_envs):
            if dones[i]:
                self.current_ids[i] = self.current_ids.max() + 1
        
        self.current_size += n_changes

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

    # def store(self, message_encodings, states):
    #     """
    #     Store step.
    #     :param message_encodings: (np.ndarray) Encoded messages, 
    #         dim=(n_parallel_envs, context_dim).
    #     :param states: (np.ndarray) Actual states, dim=(n_parallel_envs, 
    #         state_dim).
    #     """
    #     add_len = message_encodings.shape[0]
    #     if self.current_size + add_len > self.max_size:
    #         n_shift = add_len + self.current_size - self.max_size
    #         self.message_enc_buffer = np.roll(
    #             self.message_enc_buffer, -n_shift, axis=0)
    #         self.state_buffer = np.roll(self.state_buffer, -n_shift, axis=0)
    #         # print(self.message_enc_buffer)
    #         self.current_size = self.max_size
    #     else:
    #         self.current_size += add_len

    #     self.message_enc_buffer[self.current_size - add_len:self.current_size] = \
    #         message_encodings
    #     self.state_buffer[self.current_size - add_len:self.current_size] = \
    #         states

    def sample(self):
        """
        Sample a batch of training data.
        :return message_encodings: (np.ndarray) Encoded messages, 
            dim=(batch_size, context_dim).
        :return states: (np.ndarray) Actual states, dim=(batch_size, 
            state_dim).
        """
        batch_size = self.batch_size if self.batch_size <= self.current_size \
                        else self.current_size

        # TODO Adapter le sample pour que ça marche
        sample_ids = np.random.choice(
            self.current_size, batch_size, replace=False)
            
        return self.message_enc_buffer[sample_ids], self.state_buffer[sample_ids]


class SharedMemory():

    def __init__(self, args, state_dim, lang_learner, device="cpu"):
        self.hidden_dim = args.shared_mem_hidden_dim
        self.n_rec_layers = args.shared_mem_n_rec_layers
        self.n_parallel_envs = args.n_parallel_envs
        self.state_dim = state_dim
        self.device = device

        # # Language encoder
        # self.lang_learner = lang_learner

        # GRU layer
        self.gru = nn.GRU(
            args.context_dim, self.hidden_dim, self.n_rec_layers)
        init_rnn_params(self.gru)
        self.norm = nn.LayerNorm(self.hidden_dim)

        # Output prediction layer
        self.out = MLPNetwork(self.hidden_dim, state_dim, self.hidden_dim)

        # Optimizer
        self.optim = torch.optim.Adam(
            list(self.gru.parameters()) + list(self.out.parameters()), 
            lr=args.shared_mem_lr)

        # Buffer
        self.buffer = SharedMemoryBuffer(args, state_dim)

        self.memory_context = torch.zeros(
            self.n_rec_layers, self.n_parallel_envs, self.hidden_dim)

    def prep_rollout(self, device=None):
        if device is None:
            device = self.device
        else:
            self.device = device
        self.gru.eval()
        self.gru.to(device)
        self.out.eval()
        self.out.to(device)

    def prep_training(self, device=None):
        if device is None:
            device = self.device
        else:
            self.device = device
        self.gru.train()
        self.gru.to(self.device)
        self.out.train()
        self.out.to(self.device)

    def reset_context(self, env_dones=None, n_envs=None):
        if env_dones is None:
            if n_envs is None:
                n_envs = self.n_parallel_envs
            self.memory_context = torch.zeros(
                self.n_rec_layers, n_envs, self.hidden_dim).to(self.device)
        else:
            self.memory_context = self.memory_context * (1 - env_dones).astype(
                np.float32).reshape((1, self.n_parallel_envs, 1))
        
            # Change place to store in buffer for finished episode
            self.buffer.end_episode(env_dones)

    def _predict_states(self, message_encodings):
        """
        Predict states from messages.
        :param message_encodings: (np.ndarray) Encoded messages, 
            dim=(n_parallel_envs, context_dim).

        :return pred_states: (np.ndarray) Actual states, dim=(n_parallel_envs, 
            state_dim).
        """
        message_encodings = torch.Tensor(
            message_encodings).unsqueeze(0).to(self.device)
        x, self.memory_context = self.gru(
            message_encodings, self.memory_context)
        x = self.norm(x)

        pred_states = self.out(x.squeeze(0))
        return pred_states

    @torch.no_grad()
    def get_prediction_error(self, message_encodings, states):
        """
        Return the prediction error of the model given communicated messages
        and actual states.
        :param message_encodings: (np.ndarray) Encoded messages, 
            dim=(n_parallel_envs, context_dim).
        :param states: (np.ndarray) Actual states, dim=(n_parallel_envs, 
            state_dim).

        :return pred_error: (np.ndarray) Prediction error, 
            dim=(n_parallel_envs,).
        """
        self.buffer.store(message_encodings, states)

        pred_states = self._predict_states(message_encodings)

        error = np.linalg.norm(
            torch2numpy(pred_states) - states, 2, axis=-1)
        
        return error

    # def store_step(self, message_encodings, states):
    #     """
    #     Store communicated messages and actual states for later training.
    #     :param message_encodings: (np.ndarray) Encoded messages, 
    #         dim=(n_parallel_envs, context_dim).
    #     :param states: (np.ndarray) Actual states, dim=(n_parallel_envs, 
    #         state_dim).
    #     """
    #     pass

    def train(self):
        """
        Train the model.
        :return loss: (float) Training loss.
        """
        # message_encodings, states = self.buffer.sample()


        # exit()
