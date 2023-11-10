import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(
        self, input_dim, embedding_dim, hidden_dim, layer_dim, output_dim, max_len
    ):
        super(LSTMModel, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.layer_dim = layer_dim

        # Embedding layer converts integer sequences to vector sequences
        self.embedding = nn.Embedding(input_dim, embedding_dim)

        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim, num_layers=layer_dim, batch_first=True
        )

        self.max_len = max_len
        # Readout layer
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x, x_lengths):
        out = self.embedding(x)

        # Thanks to packing, LSTM don't see padding tokens
        # and this makes our model better
        out = nn.utils.rnn.pack_padded_sequence(
            out, x_lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        out, (hidden_state, cell_state) = self.lstm(out)

        # Concatenating the final forward and backward hidden states
        hidden = torch.cat((hidden_state[-2, :, :], hidden_state[-1, :, :]), dim=1)

        return self.fc(hidden)
