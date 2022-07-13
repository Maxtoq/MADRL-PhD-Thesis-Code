import torch
from torch import nn


class CommunicationPolicy(nn.Module):

    def __init__(self, context_dim, hidden_dim, merge_fn='concat'):
        super(CommunicationPolicy, self).__init__()
        self.merge_fn = merge_fn
        if self.merge_fn == 'concat':
            input_dim = 2 * context_dim
        else:
            print("ERROR in CommunicationPolicy: bad merge_fn, must be in ['concat']")
            exit()
        # Model
        self.gru = nn.GRU(
            input_dim, 
            hidden_dim,
            batch_first=True)
        # Output layer
        self.out = nn.Sequential(
            nn.Linear(hidden_dim, context_dim)
        )

    def forward(self, obs_context, lang_context, hidden_state=None):
        if obs_context.dim() == 2:
            obs_context = obs_context.unsqueeze(1)
        if self.merge_fn == 'concat':
            model_input = torch.cat((obs_context, lang_context), dim=2)
        output, hidden = self.gru(model_input, hidden_state)
        output = self.out(output.squeeze(1))
        return output, hidden
