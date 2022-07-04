import torch
from torch import nn


class CommunicationPolicy(nn.Module):

    def __init__(self, context_dim, hidden_dim):
        super(CommunicationPolicy, self).__init__()
        self.model = nn.Sequential(
            nn.GRU(
                2 * context_dim, 
                hidden_dim,
                batch_first=True),
            nn.ReLU(),
            nn.Linear(hidden_dim, context_dim)
        )

    def forward(self, internal_context, external_context):
        model_input = torch.cat((internal_context, external_context), dim=1)
        output = self.model(model_input)
        return output
