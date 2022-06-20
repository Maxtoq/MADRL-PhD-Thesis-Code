import numpy as np
import torch

from torch import nn


class OneHotEncoder(nn.Module):
    """
    Class managing the vocabulary and its one-hot encodings
    """
    def __init__(self, vocab):
        """
        Inputs:
            :param vocab (list): List of tokens that can appear in the language
        """
        self.vocab = {}
        dim_vocab = len(vocab)
        for i, w in enumerate(vocab):
            self.vocab[w] = np.eye(dim_vocab)[i]

    def get_onehots(self, sentence):
        """
        Transforms a sentence into a list of corresponding one-hot encodings
        Inputs:
            :param sentence (list): Input sentence, made of a list of tokens
        Outputs:
            :param onehots (list): List of one-hot encodings
        """
        onehots = [self.vocab[token] for token in sentence]
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
        return len(self.vocab)

class GRUEncoder(nn.Module):
    """
    Class for a language encoder using a Gated Recurrent Unit network
    """
    def __init__(self, context_dim, word_encoder, n_layers=1):
        super(GRUEncoder, self).__init__()
        self.word_encoder = word_encoder
        self.gru = nn.GRU(
            self.word_encoder.get_encoding_dim(), 
            context_dim, 
            n_layers,
            batch_first=True)

    def forward(self, sentence_batch):
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
            padded, lens, batch_first=True)

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
    def __init__(self, context_dim, word_encoder, n_layers=1):
        super(GRUDecoder, self).__init__()
        self.word_encoder = word_encoder
        self.gru = nn.GRU(
            self.word_encoder.get_encoding_dim(), 
            context_dim, 
            n_layers,
            batch_first=True)
        # Output layer
        self.out = nn.Sequential(
            nn.Linear(context_dim, self.word_encoder.get_encoding_dim()),
            nn.Softmax()
        )