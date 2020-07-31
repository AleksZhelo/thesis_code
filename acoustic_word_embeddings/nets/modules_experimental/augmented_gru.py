from typing import Optional

import numpy as np
import torch

from acoustic_word_embeddings.nets.common import get_dropout_mask


class AugmentedGRU(torch.nn.Module):

    def __init__(self, input_size, hidden_size, bias=True, dropout=0.0, dropout_hidden=False,
                 go_backward=False, batch_first=False, reverse_output=False):
        super(AugmentedGRU, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.go_backward = go_backward
        self.batch_first = batch_first
        self.dropout = dropout
        self.dropout_hidden = dropout_hidden
        self.reverse_output = reverse_output

        self.cell = torch.nn.GRUCell(input_size, hidden_size, bias=bias)

    def forward(self, x, batch_lengths, initial_state: Optional[torch.Tensor] = None):
        batch_size = x.size()[0 if self.batch_first else 1]
        total_timesteps = x.size()[1 if self.batch_first else 0]

        if initial_state is not None:
            expected_size = (batch_size, self.hidden_size)
            if initial_state.size() != expected_size:
                raise RuntimeError("Hidden state size: {0}, expected {1}".format(initial_state.size(), expected_size))
            hidden_states = initial_state.view(self.num_layers, batch_size, self.hidden_size) \
                .split(1, 0)
        else:
            hidden_states = x.new_zeros(batch_size, self.hidden_size, requires_grad=False)

        output = x.new_zeros(total_timesteps, batch_size, self.hidden_size)

        # dropout_mask_input = get_dropout_mask(self.dropout, x[0], self.training)
        dropout_mask_input = None
        dropout_mask_output = get_dropout_mask(self.dropout, output[0], self.training)
        dropout_mask_hidden = get_dropout_mask(self.dropout, hidden_states,
                                               self.training) if self.dropout_hidden else None

        max_length = batch_lengths[0].item()
        last_idx = {}
        lengths = batch_lengths.numpy()[::-1]
        for length in range(max_length):
            last_idx[length] = np.nonzero(length < lengths)[0][0]

        for time_step in range(max_length):
            if not self.go_backward:
                max_idx = batch_size - last_idx[time_step]
                inp = x[:max_idx, time_step, :] if self.batch_first else x[time_step, :max_idx, :]
            else:
                max_idx = batch_size - last_idx[max_length - time_step - 1]
                inp = x[:max_idx, max_length - time_step - 1, :] if self.batch_first \
                    else x[max_length - time_step - 1, :max_idx, :]

            if dropout_mask_input is not None:
                inp = inp * dropout_mask_input[:max_idx, :]
            state_inp = hidden_states[:max_idx, :]
            state_out = self.cell(inp, state_inp)

            output[time_step if not self.reverse_output else max_length - time_step - 1, :max_idx, :] = \
                state_out if dropout_mask_output is None else state_out * dropout_mask_output[:max_idx, :]
            hidden_states = hidden_states.clone()
            # TODO: re-check implementation for hidden states
            hidden_states[:max_idx, :] = state_out if dropout_mask_hidden is None else state_out * dropout_mask_hidden[
                                                                                                   :max_idx, :]

        return output, hidden_states.view(-1, batch_size, self.hidden_size)
