import numpy as np
import torch

from torch import nn


def init_rnn_params(rnn, gain=1.0, bias=0.0):
    for name, param in rnn.named_parameters():
        if "bias" in name:
            nn.init.constant_(param, bias)
        elif "weight" in name:
            nn.init.orthogonal_(param, gain)


class OneHotEncoder:
    """
    Class managing the vocabulary and its one-hot encodings
    """
    def __init__(self, vocab, max_len=20):
        """
        Inputs:
            :param vocab (list): List of tokens that can appear in the language
        """
        self.tokens = ["<SOS>", "<EOS>"] + vocab
        self.enc_dim = len(self.tokens)
        self.token_encodings = np.eye(self.enc_dim)
        self.max_len = max_len

        self.SOS_ENC = self.token_encodings[0]
        self.EOS_ENC = self.token_encodings[1]

        self.SOS_ID = 0
        self.EOS_ID = 1

    def index2token(self, index):
        """
        Returns the token corresponding to the given index in the vocabulary
        Inputs:
            :param index (int)
        Outputs:
            :param token (str)
        """
        if type(index) in [list, np.ndarray]:
            return [self.tokens[i] for i in index]
        else:
            return self.tokens[index]

    def enc2token(self, encoding):
        """
        Returns the token corresponding to the given one-hot encoding.
        Inputs:
            :param encoding (numpy.array): One-hot encoding.
        Outputs:
            :param token (str): Corresponding token.
        """
        if len(encoding.shape) == 1:
            return self.tokens[np.argmax(encoding)]
        elif len(encoding.shape) == 2:
            return [self.tokens[np.argmax(enconding[i])] for i in range(encoding.shape[0])]
        else:
            raise NotImplementedError("Wrong index type")

    def get_onehots(self, sentence):
        """
        Transforms a sentence into a list of corresponding one-hot encodings
        Inputs:
            :param sentence (list): Input sentence, made of a list of tokens
        Outputs:
            :param onehots (list): List of one-hot encodings
        """
        onehots = [
            self.token_encodings[self.tokens.index(t)] 
            for t in sentence
        ]
        return onehots

    def encode_batch(self, sentence_batch, append_EOS=True, pad=False):
        """
        Encodes all sentences in the given batch
        Inputs:
            :param sentence_batch (list): List of sentences (lists of tokens)
            :param append_EOS (bool): Whether to append the End of Sentence 
                token to the sentence or not.
        Outputs: 
            :param encoded_batch (list): List of encoded sentences with onehot
                vectors for each token
        """
        encoded_batch = []
        max_sent_len = 0
        for s in sentence_batch:
            encoded = self.get_onehots(s)
            if append_EOS:
                encoded.append(self.EOS_ENC)

            encoded = np.array(encoded)

            sent_len = len(encoded)
            max_sent_len = max(max_sent_len, sent_len)

            if pad:
                padding = -np.ones((self.max_len - sent_len, self.enc_dim))
                encoded = np.concatenate((encoded, padding))

            encoded_batch.append(torch.Tensor(encoded))
        
        if pad:
            encoded_batch = torch.stack(encoded_batch)[:,:max_sent_len]

        return encoded_batch

    def decode_batch(self, onehots_batch):
        """
        Decode batch of encoded sentences
        Inputs:
            :param onehots_batch (list): List of encoded sentences (list of 
                one-hots).
        Outputs:
            :param decoded_batch (list): List of sentences.
        """
        decoded_batch = []
        for enc_sentence in onehots_batch:
            decoded_batch.append(
                [self.enc2token(enc) for enc in enc_sentence])
        return decoded_batch

class GRUEncoder(nn.Module):
    """
    Class for a language encoder using a Gated Recurrent Unit network
    """
    def __init__(self, context_dim, hidden_dim, embed_dim, word_encoder, 
                 n_layers=1, device='cpu', do_embed=True):
        """
        Inputs:
            :param context_dim (int): Dimension of the context vectors (output
                of the model).
            :param hidden_dim (int): Dimension of the hidden state of the GRU
                newtork.
            :param word_encoder (OneHotEncoder): Word encoder, associating 
                tokens with one-hot encodings
            :param n_layers (int): number of layers in the GRU (default: 1)
            :param device (str): CUDA device
        """
        super(GRUEncoder, self).__init__()
        self.device = device
        self.word_encoder = word_encoder
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim
        self.do_embed = do_embed
        
        self.embed_layer = nn.Embedding(self.word_encoder.enc_dim, embed_dim)
        
        if not self.do_embed:
            embed_dim = self.word_encoder.enc_dim
            
        self.gru = nn.GRU(
            embed_dim, 
            self.hidden_dim, 
            n_layers,
            batch_first=True)
        init_rnn_params(self.gru)
        
        self.out = nn.Linear(self.hidden_dim, context_dim)
        self.norm = nn.LayerNorm(context_dim)
        
    def embed_sentences(self, sent_batch):
        # Get one-hot encodings
        enc_sent_batch = self.word_encoder.encode_batch(sent_batch)
        
        # Embed
        if self.do_embed:
            enc_ids_batch = [s.argmax(-1) for s in enc_sent_batch]
            return [self.embed_layer(s) for s in enc_ids_batch]
        else:
            return enc_sent_batch

    def forward(self, sent_batch):
        """
        Transforms sentences into embeddings
        Inputs:
            :param sentence_batch (list(list(str))): Batch of sentences.
        Outputs:
            :param unsorted_hstates (torch.Tensor): Final hidden states
                corresponding to each given sentence, dim=(1, batch_size, 
                context_dim)
        """
        # Get one-hot encodings
        enc_sent_batch = self.word_encoder.encode_batch(sent_batch)

        # Get order of sententes sorted by length decreasing
        ids = sorted(
            range(len(enc_sent_batch)), 
            key=lambda x: len(enc_sent_batch[x]), 
            reverse=True)

        # Sort the sentences by length
        sorted_list = [enc_sent_batch[i] for i in ids]

        # Embed
        if self.do_embed:
            enc_ids_batch = [s.argmax(-1) for s in sorted_list]
            model_input = [
                self.embed_layer(s.to(self.device)) for s in enc_ids_batch]
        else:
            model_input = sorted_list

        # Pad sentences
        padded = nn.utils.rnn.pad_sequence(
            model_input, batch_first=True)

        # Pack padded sentences (to not care about padded tokens)
        lens = [len(s) for s in sorted_list]
        packed = nn.utils.rnn.pack_padded_sequence(
            padded, lens, batch_first=True).to(self.device)

        # Initial hidden state
        hidden = torch.zeros(1, len(enc_sent_batch), self.hidden_dim, 
                        device=self.device)
        
        # Pass sentences into GRU model
        _, hidden_states = self.gru(packed, hidden)

        # Re-order hidden states
        unsorted_hstates = torch.zeros_like(hidden_states).to(self.device)
        unsorted_hstates[0,ids,:] = hidden_states[0,:,:]

        return self.norm(self.out(unsorted_hstates))

    def get_params(self):
        return {'gru': self.gru.state_dict(),
                'out': self.out.state_dict()}

class GRUDecoder(nn.Module):
    """
    Class for a language decoder using a Gated Recurrent Unit network
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
        super(GRUDecoder, self).__init__()
        self.device = device
        # Dimension of hidden states
        self.hidden_dim = context_dim
        # Word encoder
        self.word_encoder = word_encoder
        # Number of recurrent layers
        self.n_layers = n_layers
        # Model
        self.gru = nn.GRU(
            self.word_encoder.enc_dim, 
            self.hidden_dim, 
            self.n_layers)
        init_rnn_params(self.gru)
        # Output layer
        self.out = nn.Sequential(
            nn.Linear(self.hidden_dim, self.word_encoder.enc_dim),
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
    
    def forward(self, context_batch, target_encs=None):
        """
        Transforms context vectors to sentences
        Inputs:
            :param context_batch (torch.Tensor): Batch of context vectors,
                dim=(batch_size, context_dim).
            :param target_encs (torch.Tensor): Batch of target sentences used
                for teacher forcing, encoded as onehots and padded with -1, 
                dim=(batch_size, max_sent_len, enc_dim). If None then no 
                teacher forcing. Default: None.
        Outputs:
            :param decoder_outputs (list): Batch of tensors containing
                log-probabilities generated by the GRU network.
            :param sentences (list): Sentences generated with greedy 
                sampling. Empty if target_encs is not None (teacher forcing,
                so we only care about model predictions).
        """
        teacher_forcing = target_encs is not None
        batch_size = context_batch.size(0)
        max_sent_len = target_encs.shape[1] if teacher_forcing \
            else self.word_encoder.max_len

        hidden = context_batch.unsqueeze(0)
        last_tokens = torch.Tensor(self.word_encoder.SOS_ENC).view(
            1, 1, -1).float().repeat(1, batch_size, 1).to(self.device)

        tokens = []
        decoder_outputs = []
        sentences = [[] for b_i in range(batch_size)]
        sent_finished = [False] * batch_size
        for t_i in range(max_sent_len):
            # RNN pass
            outputs, hidden = self.forward_step(last_tokens, hidden)
            decoder_outputs.append(outputs)

            # Sample next tokens
            if teacher_forcing:
                last_tokens = target_encs[:, t_i].unsqueeze(0)
            else:
                _, topi = outputs.topk(1)
                topi = topi.squeeze()
                last_tokens = torch.Tensor(
                    self.word_encoder.token_encodings[topi.cpu()]).unsqueeze(0).to(
                        self.device)

                for b_i in range(batch_size):
                    if topi[b_i] == self.word_encoder.EOS_ID:
                        sent_finished[b_i] = True
                    if not sent_finished[b_i]:
                        sentences[b_i].append(
                            self.word_encoder.index2token(topi[b_i]))
                
                if all(sent_finished):
                    break
                    
        decoder_outputs = torch.cat(decoder_outputs, axis=0).transpose(0, 1)

        return decoder_outputs, sentences

    def compute_pp(self, enc_sent):
        """
        :param enc_sent: (list(torch.Tensor))
        """
        batch_size = len(enc_sent)
        max_sent_len = max([len(s) for s in enc_sent])

        hidden = torch.zeros((self.n_layers, batch_size, self.hidden_dim))
        last_tokens = torch.Tensor(self.word_encoder.SOS_ENC).view(
            1, 1, -1).repeat(1, batch_size, 1).to(self.device)

        pnorm = torch.ones(batch_size)
        for t_i in range(max_sent_len):
            # RNN pass
            outputs, hidden = self.forward_step(last_tokens, hidden)

            # Compute PP
            probs = outputs.exp().squeeze(0)
            for s_i in range(batch_size):
                len_s = enc_sent[s_i].size(0)
                if t_i < len_s:
                    token_prob = (probs[s_i] * enc_sent[s_i][t_i]).sum(-1)
                    pnorm[s_i] *= (token_prob ** (1 / len_s))

            # Do teacher forcing
            last_tokens = torch.zeros_like(last_tokens).to(self.device)
            for s_i in range(batch_size):
                if t_i < enc_sent[s_i].size(0):
                    last_tokens[0, s_i] = enc_sent[s_i][t_i]

        pp = 1 / pnorm

        return pp

    def get_params(self):
        return {'gru': self.gru.state_dict(),
                'out': self.out.state_dict()}
                    