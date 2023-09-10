import torch
from torch import nn

from .lm import OneHotEncoder, GRUEncoder, GRUDecoder
from .obs import ObservationEncoder
from .lang_buffer import LanguageBuffer
    

class LanguageLearner:

    """ 
    Class to manage and train the language modules: the Language Encoder, the 
    Observation Encoder and the Decoder. 
    """

    def __init__(self, obs_dim, context_dim, hidden_dim, vocab, device, 
                 lr=0.007, n_epochs=2, batch_size=128, temp=1.0, 
                 clip_weight=1.0, capt_weight=1.0, obs_learn_capt=True, 
                 buffer_size=100000):
        self.train_device = device
        self.lr = lr
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.temp = temp
        self.clip_weight = clip_weight
        self.capt_weight = capt_weight
        self.obs_learn_capt = obs_learn_capt

        self.word_encoder = OneHotEncoder(vocab)

        self.obs_encoder = ObservationEncoder(obs_dim, context_dim, hidden_dim)
        self.lang_encoder = GRUEncoder(context_dim, hidden_dim, self.word_encoder)
        self.decoder = GRUDecoder(context_dim, self.word_encoder)

        self.clip_loss = nn.CrossEntropyLoss()
        self.captioning_loss = nn.NLLLoss()

        self.optim = torch.optim.Adam(
            list(self.obs_encoder.parameters()) + 
            list(self.lang_encoder.parameters()) + 
            list(self.decoder.parameters()), 
            lr=lr)

        self.buffer = LanguageBuffer(buffer_size)

    def prep_rollout(self, device=None):
        if device is None:
            device = self.train_device
        self.obs_encoder.eval()
        self.obs_encoder.to(device)
        self.lang_encoder.eval()
        self.lang_encoder.to(device)
        self.decoder.eval()
        self.decoder.to(device)

    def prep_training(self):
        self.obs_encoder.train()
        self.obs_encoder.to(self.train_device)
        self.lang_encoder.train()
        self.lang_encoder.to(self.train_device)
        self.decoder.train()
        self.decoder.to(self.train_device)
    
    def store(self, obs, sent):
        self.buffer.store(obs, sent)

    def encode_sentences(self, sentence_batch):
        """ 
        Encode a batch of sentences. 
        :param sentence_batch (list(list(str))): Batch of sentences.

        :return context_batch (torch.Tensor): Batch of context vectors, 
            dim=(batch_size, context_dim).
        """
        context_batch = self.lang_encoder(sentence_batch).squeeze(0)
        return context_batch
    
    def encode_observations(self, obs_batch):
        context_batch = self.obs_encoder(obs_batch)
        return context_batch

    def generate_sentences(self, context_batch):
        """ 
        Generate sentences from a batch of context vectors. 
        :param context_batch (torch.Tensor): Batch of context vectors,
            dim=(batch_size, context_dim).
        
        :return gen_sent_batch (list(list(str))): Batch of generated sentences.
        """
        _, sentences = self.decoder(context_batch)
        return sentences

    def compute_losses(self, obs_batch, sent_batch):
        # Encode observations
        obs_tensor = torch.Tensor(np.array(obs_batch))
        obs_context_batch = self.obs_encoder(obs_tensor)

        # Encode sentences
        lang_context_batch = self.lang_encoder(sent_batch)
        lang_context_batch = lang_context_batch.squeeze()

        # Compute similarity
        norm_context_batch = obs_context_batch / obs_context_batch.norm(
            dim=1, keepdim=True)
        lang_context_batch = lang_context_batch / lang_context_batch.norm(
            dim=1, keepdim=True)
        sim = norm_context_batch @ lang_context_batch.t() * self.temp
        mean_sim = sim.diag().mean()

        # Compute CLIP loss
        labels = torch.arange(len(obs_batch))
        loss_o = self.clip_loss(sim, labels)
        loss_l = self.clip_loss(sim.t(), labels)
        clip_loss = (loss_o + loss_l) / 2
        
        # Decoding
        encoded_targets = self.word_encoder.encode_batch(sent_batch)
        if not self.obs_learn_capt:
            obs_context_batch = obs_context_batch.detach()
        decoder_outputs, _ = self.decoder(obs_context_batch, encoded_targets)

        # Compute Captioning loss
        dec_loss = 0
        for d_o, e_t in zip(decoder_outputs, encoded_targets):
            e_t = torch.argmax(e_t, dim=1)
            dec_loss += self.captioning_loss(d_o[:e_t.size(0)], e_t)
        
        return clip_loss, dec_loss, mean_sim

    def train(self):
        clip_losses = []
        dec_losses = []
        mean_sims = []
        for it in range(self.n_epochs):
            self.optim.zero_grad()
            # Sample batch from buffer
            obs_batch, sent_batch = self.buffer.sample(self.batch_size)

            # Compute losses
            clip_loss, dec_loss, mean_sim = self.compute_losses(obs_batch, sent_batch)

            # Update
            tot_loss = self.clip_weight * clip_loss + self.capt_weight * dec_loss
            tot_loss.backward()
            self.optim.step()

            clip_losses.append(clip_loss.item() / batch_size)
            dec_losses.append(dec_loss.item() / batch_size)
            mean_sims.append(mean_sim.item())
        
        
        clip_loss = sum(clip_losses) / len(clip_losses)
        dec_loss = sum(dec_losses) / len(dec_losses)
        mean_sim = sum(mean_sims) / len(mean_sims)
        
        return clip_loss, dec_loss, mean_sim

    def get_save_dict(self):
        save_dict = {
            "obs_encoder": self.obs_encoder.state_dict(),
            "lang_encoder": self.lang_encoder.state_dict(),
            "decoder": self.decoder.state_dict(),
            "optim": self.optim.state_dict()}
        return save_dict