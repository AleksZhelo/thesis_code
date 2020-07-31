import enum
from typing import Optional

import torch
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence

from acoustic_word_embeddings.core.loss.decov_reg_loss import decov_2d, decov_3d
from acoustic_word_embeddings.nets.modules_experimental.augmented_gru import AugmentedGRU


class BiRNNMode(enum.Enum):
    # this was not really a good idea, leave at CONCAT and forget about it
    CONCAT = 1
    INTERLEAVE = 2


class CustomGRU(torch.nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, bias=True, dropout=0.0, dropout_last=False,
                 dropout_hidden=False, bidirectional=False, batch_first=False, mode=BiRNNMode.CONCAT,
                 reverse_backward_output=False, decov_output=None, decov_hidden=None):
        super(CustomGRU, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.batch_first = batch_first
        self.mode = mode
        self.decov_output = decov_output if decov_output is not None else []
        self.decov_hidden = decov_hidden if decov_hidden is not None else []

        self.layers_forward = []
        self.layers_backward = []
        for layer_idx in range(num_layers):
            if layer_idx == num_layers - 1 and not dropout_last:
                dropout = 0
            forward_layer = AugmentedGRU(input_size, hidden_size, bias=bias, dropout=dropout,
                                         dropout_hidden=dropout_hidden)
            backward_layer = AugmentedGRU(input_size, hidden_size, bias=bias, dropout=dropout,
                                          dropout_hidden=dropout_hidden, go_backward=True,
                                          reverse_output=reverse_backward_output) if bidirectional else None

            input_size = hidden_size * 2
            self.add_module('forward_layer_{}'.format(layer_idx), forward_layer)
            self.add_module('backward_layer_{}'.format(layer_idx), backward_layer)
            self.layers_forward.append(forward_layer)
            self.layers_backward.append(backward_layer)
        self.loss_extra = 0

    def forward(self, inputs: PackedSequence, initial_state: Optional[torch.Tensor] = None):
        self.loss_extra = 0
        x, batch_lengths = pad_packed_sequence(inputs, batch_first=self.batch_first)
        num_directions = 2 if self.bidirectional else 1

        if initial_state is not None:
            # TODO: not tested
            expected_size = (self.num_layers * num_directions, -1, self.hidden_size)
            if initial_state.size() != expected_size:
                raise RuntimeError("Hidden state size: {0}, expected {1}".format(initial_state.size(), expected_size))
            initial_state = initial_state.view(self.num_layers, num_directions, -1, self.hidden_size)
            initial_state_fwd = initial_state[:, 0, :, :].split(1, 0)
            if self.bidirectional:
                initial_state_bwd = initial_state[:, 1, :, :].split(1, 0)
        else:
            initial_state_fwd = [None] * self.num_layers
            if self.bidirectional:
                initial_state_bwd = [None] * self.num_layers

        outputs = x
        final_states = []
        for layer_idx in range(self.num_layers):
            x_fwd, h_fwd = self.layers_forward[layer_idx](outputs, batch_lengths, initial_state_fwd[layer_idx])
            if self.bidirectional:
                x_bwd, h_bwd = self.layers_backward[layer_idx](outputs, batch_lengths, initial_state_bwd[layer_idx])

            if self.mode == BiRNNMode.CONCAT:
                outputs = torch.cat((x_fwd, x_bwd), -1)
            elif self.mode == BiRNNMode.INTERLEAVE:
                outputs = torch.stack((x_fwd, x_bwd), -1).view(x_fwd.size()[0], x_fwd.size()[1], -1)
            final_states.append(torch.cat((h_fwd, h_bwd), 0))

            if layer_idx < len(self.decov_hidden):
                weight_hidden = self.decov_hidden[layer_idx]
                if weight_hidden > 0:
                    self.loss_extra += weight_hidden * decov_2d(h_fwd.squeeze())
                    if self.bidirectional:
                        self.loss_extra += weight_hidden * decov_2d(h_bwd.squeeze())

            if layer_idx < len(self.decov_output):
                weight_output = self.decov_output[layer_idx]
                if weight_output > 0:
                    self.loss_extra += weight_output * decov_3d(x_fwd, batch_lengths)
                    if self.bidirectional:
                        self.loss_extra += weight_output * decov_3d(x_bwd, batch_lengths)

        return outputs, torch.cat(final_states, 0)
