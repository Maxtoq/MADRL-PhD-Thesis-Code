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

    def forward(self, internal_context, external_context, hidden_state=None):
        if self.merge_fn == 'concat':
            model_input = torch.cat((internal_context, external_context), dim=1)
        output, hidden = self.gru(model_input, hidden_state)
        output = self.out(output)
        return output, hidden
