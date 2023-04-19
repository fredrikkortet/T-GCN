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


def calculate_laplacian_with_self_loop(matrix):
    matrix = matrix + torch.eye(matrix.size(0))
    row_sum = matrix.sum(1)
    d_inv_sqrt = torch.pow(row_sum, -0.5).flatten()
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    normalized_laplacian = (
        matrix.matmul(d_mat_inv_sqrt).transpose(0, 1).matmul(d_mat_inv_sqrt)
    )
    return normalized_laplacian

def get_attention_adj_matrix(adj_matrix):
    dims = adj_matrix.shape[0]
    adj_tensor = torch.tensor(adj_matrix, dtype=torch.float32)
    self_attention = SelfAttention(input_dim=dims, hidden_dim=dims)
    output = self_attention(adj_tensor)
    output = output.detach().numpy()
    output = calculate_laplacian_with_self_loop(torch.FloatTensor(output))
    return output