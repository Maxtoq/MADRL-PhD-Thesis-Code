import random


class LanguageBuffer:

    def __init__(self, max_steps):
        self.max_steps = max_steps

        self.obs_buffer = []
        self.sent_buffer = []

    def store(self, obs, sent_list):
        for obs, sent in zip(obs, sent_list):
            if len(self.obs_buffer) == self.max_steps:
                self.obs_buffer.pop(0)
                self.sent_buffer.pop(0)
            self.obs_buffer.append(obs)
            self.sent_buffer.append(" ".join(sent))

    def sample(self, batch_size):
        if batch_size > len(self.obs_buffer):
            batch_size = len(self.obs_buffer)
        obs_batch = []
        sent_batch = []
        nb_sampled = 0
        while nb_sampled < batch_size:
            index = random.randrange(len(self.obs_buffer))
            obs = self.obs_buffer[index]
            sent = self.sent_buffer[index]
            if sent in sent_batch:
                continue
            else:
                obs_batch.append(obs)
                sent_batch.append(sent.split(" ") if sent != '' else [])
                nb_sampled += 1
        return obs_batch, sent_batch