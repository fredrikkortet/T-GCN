import argparse
import torch
import torch.nn as nn
from utils.graph_conv import calculate_laplacian_with_self_loop


class TGCNGraphConvolution(nn.Module):
    def __init__(self, adj, num_lstm_units: int, output_dim: int, bias: float = 0.0):
        super(TGCNGraphConvolution, self).__init__()
        self._num_lstm_units = num_lstm_units
        self._output_dim = output_dim
        self._bias_init_value = bias
        self.register_buffer(
            "laplacian", calculate_laplacian_with_self_loop(torch.FloatTensor(adj))
        )
        self.weights = nn.Parameter(
            torch.FloatTensor(self._num_lstm_units + 1, self._output_dim)
        )
        self.biases = nn.Parameter(torch.FloatTensor(self._output_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weights)
        nn.init.constant_(self.biases, self._bias_init_value)

    def forward(self, inputs, hidden_state):
        batch_size, num_nodes = inputs.shape
        # inputs (batch_size, num_nodes) -> (batch_size, num_nodes, 1)
        inputs = inputs.reshape((batch_size, num_nodes, 1))
        # hidden_state (batch_size, num_nodes, num_lstm_units)
        hidden_state = hidden_state.reshape(
            (batch_size, num_nodes, self._num_lstm_units)
        )
        # [x, h] (batch_size, num_nodes, num_lstm_units + 1)
        concatenation = torch.cat((inputs, hidden_state), dim=2)
        # [x, h] (num_nodes, num_lstm_units + 1, batch_size)
        concatenation = concatenation.transpose(0, 1).transpose(1, 2)
        # [x, h] (num_nodes, (num_lstm_units + 1) * batch_size)
        concatenation = concatenation.reshape(
            (num_nodes, (self._num_lstm_units + 1) * batch_size)
        )
        # A[x, h] (num_nodes, (num_lstm_units + 1) * batch_size)
        a_times_concat = self.laplacian @ concatenation
        # A[x, h] (num_nodes, num_lstm_units + 1, batch_size)
        a_times_concat = a_times_concat.reshape(
            (num_nodes, self._num_lstm_units + 1, batch_size)
        )
        # A[x, h] (batch_size, num_nodes, num_lstm_units + 1)
        a_times_concat = a_times_concat.transpose(0, 2).transpose(1, 2)
        # A[x, h] (batch_size * num_nodes, num_lstm_units + 1)
        a_times_concat = a_times_concat.reshape(
            (batch_size * num_nodes, self._num_lstm_units + 1)
        )
        # A[x, h]W + b (batch_size * num_nodes, output_dim)
        outputs = a_times_concat @ self.weights + self.biases
        # A[x, h]W + b (batch_size, num_nodes, output_dim)
        outputs = outputs.reshape((batch_size, num_nodes, self._output_dim))
        # A[x, h]W + b (batch_size, num_nodes * output_dim)
        outputs = outputs.reshape((batch_size, num_nodes * self._output_dim))
        return outputs

    @property
    def hyperparameters(self):
        return {
            "num_ugrnn_units": self._num_lstm_units,
            "output_dim": self._output_dim,
            "bias_init_value": self._bias_init_value,
        }


class TGCNCell(nn.Module):
    def __init__(self, adj, input_dim: int, hidden_dim: int):
        super(TGCNCell, self).__init__()
        self._input_dim = input_dim
        self._hidden_dim = hidden_dim
        self.register_buffer("adj", torch.FloatTensor(adj))
        self.graph_conv1 = TGCNGraphConvolution(
            self.adj, self._hidden_dim, self._hidden_dim * 2, bias=1.0
        )
        self.graph_conv2 = TGCNGraphConvolution(
            self.adj, self._hidden_dim, self._hidden_dim
        )

    def forward(self, inputs, hidden_state):
        # [r, u] = sigmoid(A[x, h]W + b)
        # [r, u] (batch_size, num_nodes * (3 * num_lstm_units))
        concatenation = torch.sigmoid(self.graph_conv1(inputs, hidden_state))
        # r (batch_size, num_nodes, num_ugrnn_units)
        # u (batch_size, num_nodes, num_ugrnn_units)
        r, u = torch.chunk(concatenation, chunks=2, dim=1)
        #h_t = (1 - u) * h + u * tanh(A[x, h]W + b)
        new_hidden_state = (1 - u) * hidden_state + u * torch.tanh(self.graph_conv1(inputs, hidden_state))
        
        return new_hidden_state, new_hidden_state

    @property
    def hyperparameters(self):
        return {"input_dim": self._input_dim, "hidden_dim": self._hidden_dim}


class TGCN_UGRNN(nn.Module):
    def __init__(self, adj, hidden_dim: int,dropout: float, layer_2: bool, **kwargs):
        super(TGCN_UGRNN, self).__init__()
        self._input_dim = adj.shape[0]
        self._hidden_dim = hidden_dim
        self._layer_2 = layer_2
        self.register_buffer("adj", torch.FloatTensor(adj))
        self.tgcn_cell = TGCNCell(self.adj, self._input_dim, self._hidden_dim)
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(self.dropout)

    def forward(self, inputs):
        batch_size, seq_len, num_nodes = inputs.shape
        assert self._input_dim == num_nodes
        hidden_state = torch.zeros(batch_size, num_nodes * self._hidden_dim).type_as(
            inputs
        )
        cell_state = torch.zeros(batch_size, num_nodes * self._hidden_dim).type_as(
            inputs
        )
        output = None
        for i in range(seq_len):
            hidden_state = self.dropout_layer(hidden_state)
            output, hidden_state = self.tgcn_cell(inputs[:, i, :], hidden_state)
            if self._layer_2:
                output, hidden_state = self.tgcn_cell(inputs[:, i, :], hidden_state)
            
            output = output.reshape((batch_size, num_nodes, self._hidden_dim))
        return output

    @staticmethod
    def add_model_specific_arguments(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--hidden_dim", type=int, default=64)
        parser.add_argument("--dropout", type=float, default=0)
        parser.add_argument("--cell_dim", type=int, default=64)
        parser.add_argument("--layer_2", type=bool, default=False)
        return parser

    @property
    def hyperparameters(self):
        return {"input_dim": self._input_dim, "hidden_dim": self._hidden_dim}
