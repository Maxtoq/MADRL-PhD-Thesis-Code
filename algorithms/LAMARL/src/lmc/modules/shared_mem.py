import numpy as np
import torch

from torch import nn

from .lm import init_rnn_params
from .networks import MLPNetwork
from src.lmc.utils import torch2numpy


class SharedMemoryBuffer():

    def __init__(self, args):
        self.n_parallel_envs = args.n_parallel_envs
        self.max_size = args.shared_mem_max_buffer_size
        self.batch_size = args.shared_mem_batch_size

        self.message_enc_buffer = []
        self.state_buffer = []

    def store(self, message_encodings, states):
        """
        Return the prediction error of the model given communicated messages
        and actual states.
        :param message_encodings: (np.ndarray) Encoded messages, 
            dim=(batch_size, context_dim).
        :param states: (np.ndarray) Actual states, dim=(batch_size, 
            state_dim).
        """
        pass

    def sample(self):
        pass


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
        self.buffer = SharedMemoryBuffer(args)

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
        if n_envs is None:
            n_envs = self.n_parallel_envs
        if env_dones is None:
            self.memory_context = torch.zeros(
                self.n_rec_layers, n_envs, self.hidden_dim).to(self.device)
        else:
            print(self.memory_context.shape, env_dones.shape)
            exit()
            self.memory_context = self.memory_context * (1 - env_dones).astype(
                np.float32)[..., np.newaxis]

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
            dim=(batch_size, context_dim).
        :param states: (np.ndarray) Actual states, dim=(batch_size, 
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
        pass
