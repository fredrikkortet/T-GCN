import argparse
import torch
import torch.nn as nn


class LSTMLinear(nn.Module):
    def __init__(self, num_lstm_units: int, output_dim: int, bias: float = 0.0):
        super(LSTMLinear, self).__init__()
        self._num_lstm_units = num_lstm_units
        self._output_dim = output_dim
        self._bias_init_value = bias
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
        # inputs (batch_size, num_nodes, 1)
        inputs = inputs.reshape((batch_size, num_nodes, 1))
        # hidden_state (batch_size, num_nodes, num_lstm_units)
        hidden_state = hidden_state.reshape(
            (batch_size, num_nodes, self._num_lstm_units)
        )
        # [inputs, hidden_state] "[x, h]" (batch_size, num_nodes, num_lstm_units + 1)
        concatenation = torch.cat((inputs, hidden_state), dim=2)
        # [x, h] (batch_size * num_nodes, gru_units + 1)
        concatenation = concatenation.reshape((-1, self._num_lstm_units + 1))
        # [x, h]W + b (batch_size * num_nodes, output_dim)
        outputs = concatenation @ self.weights + self.biases
        # [x, h]W + b (batch_size, num_nodes, output_dim)
        outputs = outputs.reshape((batch_size, num_nodes, self._output_dim))
        # [x, h]W + b (batch_size, num_nodes * output_dim)
        outputs = outputs.reshape((batch_size, num_nodes * self._output_dim))
        return outputs

    def hyperparameters(self):
        return {
            "num_lstm_units": self._num_lstm_units,
            "output_dim": self._output_dim,
            "bias_init_value": self._bias_init_value,
        }


class LSTMCell(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, cell_dim: int):
        super(LSTMCell, self).__init__()
        self._input_dim = input_dim
        self._hidden_dim = hidden_dim
        self._cell_dim = cell_dim
        self.linear1 = LSTMLinear(self._hidden_dim, self._hidden_dim * 3, bias=1.0)
        self.linear2 = LSTMLinear(self._hidden_dim, self._hidden_dim)

    def forward(self, inputs, hidden_state, cell):
        # [r, u] = sigmoid([x, h]W + b)
        # [r, u] (batch_size, num_nodes * (2 * num_lstm_units))
        concatenation = torch.sigmoid(self.linear1(inputs, hidden_state))
        # r (batch_size, num_nodes * num_lstm_units)
        # u (batch_size, num_nodes * num_lstm_units)
        i, f, o = torch.chunk(concatenation, chunks=3, dim=1)
        # c = tanh([x, (r * h)]W + b)
        # c (batch_size, num_nodes * num_lstm_units)
        c = torch.tanh(self.linear2(inputs, hidden_state))
        # h := u * h + (1 - u) * c
        # h (batch_size, num_nodes * num_lstm_units)
        cell = f * cell + i * c
        
        new_hidden_state = o * torch.tanh(cell)
        
        return new_hidden_state, new_hidden_state, cell

    @property
    def hyperparameters(self):
        return {"input_dim": self._input_dim, "hidden_dim": self._hidden_dim}


class LSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, cell_dim: int, **kwargs):
        super(LSTM, self).__init__()
        self._input_dim = input_dim  # num_nodes for prediction
        self._hidden_dim = hidden_dim
        self._cell_dim = cell_dim
        self.lstm_cell = LSTMCell(self._input_dim, self._hidden_dim, self._cell_dim)

    def forward(self, inputs):
        batch_size, seq_len, num_nodes = inputs.shape
        assert self._input_dim == num_nodes
        outputs = list()
        hidden_state = torch.zeros(batch_size, num_nodes * self._hidden_dim).type_as(
            inputs
        )
        cell_state = torch.zeros(batch_size, num_nodes * self._hidden_dim).type_as(
            inputs
        )
        for i in range(seq_len):
            output, hidden_state, cell_state = self.lstm_cell(inputs[:, i, :], hidden_state,cell_state)
            output = output.reshape((batch_size, num_nodes, self._hidden_dim))
            outputs.append(output)
        last_output = outputs[-1]
        return last_output

    @staticmethod
    def add_model_specific_arguments(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--hidden_dim", type=int, default=64)
        parser.add_argument("--cell_dim", type=int, default=64)
        parser.add_argument("--dropout", type=float, default=0)
        parser.add_argument("--layer_2", type=bool, default=False)
        return parser

    @property
    def hyperparameters(self):
        return {"input_dim": self._input_dim, "hidden_dim": self._hidden_dim}
