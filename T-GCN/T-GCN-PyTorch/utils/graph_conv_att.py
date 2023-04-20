import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from torch.nn.functional import normalize

class SelfAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim=None):
        super().__init__()

        if hidden_dim is None:
            hidden_dim = input_dim

        self.query = nn.Linear(input_dim, hidden_dim, bias=False)
        self.key = nn.Linear(input_dim, hidden_dim, bias=False)
        self.value = nn.Linear(input_dim, hidden_dim, bias=False)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(K.shape[-1], dtype=torch.float32))

        attention_weights = self.softmax(attention_scores)

        output = torch.matmul(attention_weights, V)

        return output

def get_attention_adj_matrix(adj_matrix):
    dims = adj_matrix.shape[0]
    adj_tensor = torch.tensor(adj_matrix, dtype=torch.float32)
    self_attention = SelfAttention(input_dim=dims, hidden_dim=dims)
    output = self_attention(adj_tensor)
    output = output.detach().numpy()
    return output