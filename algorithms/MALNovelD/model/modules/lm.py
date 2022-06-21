import numpy as np
import torch

from torch import nn


class OneHotEncoder:
    """
    Class managing the vocabulary and its one-hot encodings
    """
    def __init__(self, vocab):
        """
        Inputs:
            :param vocab (list): List of tokens that can appear in the language
        """
        self.token_indexes = ["<SOS>", "<EOS>"] + vocab
        dim_vocab = len(self.token_indexes)
        self.token_encodings = np.eye(dim_vocab)

        self.SOS_ENC = self.token_encodings[0]
        self.EOS_ENC = self.token_encodings[1]

    def index2token(self, index):
        """
        Return the token corresponding to the given index in the vocabulary
        Inputs:
            :param index (int)
        Outputs:
            :param token (str)
        """
        return self.token_indexes[index]

    def get_onehots(self, sentence):
        """
        Transforms a sentence into a list of corresponding one-hot encodings
        Inputs:
            :param sentence (list): Input sentence, made of a list of tokens
        Outputs:
            :param onehots (list): List of one-hot encodings
        """
        onehots = [
            self.token_encodings[self.token_indexes.index(t)] 
            for t in sentence
        ]
        return onehots

    def encode_batch(self, sentence_batch):
        """
        Encodes all sentences in the given batch
        Inputs:
            :param sentence_batch (list): List of sentences (lists of tokens)
        Outputs: 
            :param encoded_batch (list): List of encoded sentences as Torch 
                tensors
        """
        encoded_batch = [torch.Tensor(
            np.array(self.get_onehots(s))) for s in sentence_batch]
        return encoded_batch

    def get_encoding_dim(self):
        """
        Returns the dimension of the one-hot encodings
        Outputs:
            :param encoding_dim (int): dimension of the one-hot encodings
        """
        return len(self.token_indexes)

class GRUEncoder(nn.Module):
    """
    Class for a language encoder using a Gated Recurrent Unit network
    """
    def __init__(self, context_dim, word_encoder, n_layers=1, device='cpu'):
        """
        Inputs:
            :param context_dim (int): Dimension of the context vectors
            :param word_encoder (OneHotEncoder): Word encoder, associating 
                tokens with one-hot encodings
            :param n_layers (int): number of layers in the GRU (default: 1)
            :param device (str): CUDA device
        """
        super(GRUEncoder, self).__init__()
        self.device = device
        self.word_encoder = word_encoder
        self.gru = nn.GRU(
            self.word_encoder.get_encoding_dim(), 
            context_dim, 
            n_layers,
            batch_first=True)

    def forward(self, sentence_batch):
        """
        Transforms sentences into embeddings
        Inputs:
            :param sentence_batch (list): Batch of sentences (lists of tokens)
        Outputs:
            :param unsorted_hstates (torch.Tensor): Final hidden states
                corresponding to each given sentence
        """
        # Get one-hot encodings
        enc = self.word_encoder.encode_batch(sentence_batch)

        # Get order of sententes sorted by length decreasing
        ids = sorted(range(len(enc)), key=lambda x: len(enc[x]), reverse=True)

        # Sort the sentences by length
        sorted_list = [enc[i] for i in ids]

        # Pad sentences
        padded = nn.utils.rnn.pad_sequence(
            sorted_list, batch_first=True)

        # Pack padded sentences (to not care about padded tokens)
        lens = [len(s) for s in sorted_list]
        packed = nn.utils.rnn.pack_padded_sequence(
            padded, lens, batch_first=True).to(self.device)

        # Pass sentences into GRU model
        _, hidden_states = self.gru(packed)

        # Re-order hidden states
        unsorted_hstates = torch.zeros_like(hidden_states)
        unsorted_hstates[0,ids,:] = hidden_states[0,:,:]

        return unsorted_hstates

class GRUDecoder(nn.Module):
    """
    Class for a language decoder using a Gated Recurrent Unit network
    """
    def __init__(self, context_dim, word_encoder, n_layers=1, max_length=15,
                 device='cpu'):
        """
        Inputs:
            :param context_dim (int): Dimension of the context vectors
            :param word_encoder (OneHotEncoder): Word encoder, associating 
                tokens with one-hot encodings
            :param n_layers (int): number of layers in the GRU (default: 1)
            :param device (str): CUDA device
        """
        super(GRUDecoder, self).__init__()
        self.device = device
        # Dimension of hidden states
        self.hidden_dim = context_dim
        # Word encoder
        self.word_encoder = word_encoder
        # Max length of generated sentences
        self.max_length = max_length
        # Model
        self.gru = nn.GRU(
            self.word_encoder.get_encoding_dim(), 
            self.hidden_dim, 
            n_layers,
            batch_first=True)
        # Output layer
        self.out = nn.Sequential(
            nn.Linear(self.hidden_dim, self.word_encoder.get_encoding_dim()),
            nn.LogSoftmax(dim=2)
        )

    def forward_step(self, last_token, last_hidden):
        """
        Generate prediction from GRU network.
        Inputs:
            :param last_token (torch.Tensor): Token at last time step, 
                dim=(1, 1, token_dim).
            :param last_hidden (torch.Tensor): Hidden state of the GRU at last
                time step, dim=(1, 1, hidden_dim).
        Outputs:
            :param output (torch.Tensor): Log-probabilities outputed by the 
                model, dim=(1, 1, token_dim).
            :param hidden (torch.Tensor): New hidden state of the GRU network,
                dim=(1, 1, hidden_dim).
        """
        output, hidden = self.gru(last_token, last_hidden)
        output = self.out(output)
        return output, hidden

    def forward(self, context_batch, target_encs=None, greedy_pred=True):
        """
        Transforms context vectors to sentences
        Inputs:
            :param context_batch (torch.Tensor): Batch of context vectors,
                dim=(batch_size, context_dim).
            :param target_encs (list): Batch of target encoded sentences used
                for teacher forcing. If None then no teacher forcing. 
                (Default: None)
            :param greedy_pred (boolean): If we need sentences generated with
                greedy sampling in the output.
        Outputs:
            :param decoder_outputs (list): Batch of tensors containing
                log-probabilities generated by the GRU network.
            :param greedy_preds (list): Sentences generated with greedy 
                sampling. None if target_encs is not None (teacher forcing,
                so we only care about model predictions) or greedy_pred is 
                False.
        """
        teacher_forcing = target_encs is None
        batch_size = context_batch.size(0)

        # Initial hidden state
        hidden = torch.zeros(1, 1, self.hidden_dim, device=self.device)

        # Starting of sentence token
        decoder_input = torch.tensor(
            np.array([[self.word_encoder.SOS_ENC]]),
            device=self.device)

        # For each input vector in batch
        for b_i in range(batch_size):
            max_l = target_encs.size(1) if teacher_forcing else self.max_length
            # For each token to generate
            for t_i in range(max_l):
                # Get prediction
                output, hidden = self.forward_step(decoder_input, hidden)

                # Add output to list

                # Decode output and add to generated sentence

                # Set next decoder input

