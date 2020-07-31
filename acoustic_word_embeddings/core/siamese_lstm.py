import numpy as np
import torch

from acoustic_word_embeddings.core.loss.embedding_loss import triplet_loss_offline
from base import util
from base.settings import Settings
from acoustic_word_embeddings.nets.common import hidden2fc_input
from acoustic_word_embeddings.nets.lstm_fc_base import LSTM_FC_base


class SiameseLSTM(LSTM_FC_base):

    def __init__(self, logger, config, batch_first=False, loss=None):
        super(SiameseLSTM, self).__init__(logger, config, batch_first)
        if loss is None:
            loss = triplet_loss_offline(config, -1)
        self.loss_delegate = loss
        self.loss_delegate.register_params(self)
        util.warn_or_print(logger, 'Using {0} loss'.format(loss.__class__.__name__))

    def forward(self, x):
        return self.loss_delegate.mod_forward(super().forward(x))

    def stepwise_embeddings(self, x):
        if self.lstm.bidirectional:
            sz = x.size()
            batch = torch.zeros(sz[0], sz[0], sz[2], dtype=torch.float, device='cuda' if self.use_cuda else 'cpu')
            lengths = np.array(np.arange(sz[0])[::-1] + 1)
            for i, length in enumerate(lengths):
                batch[:length, i, :] = x[:length, 0, :]
            batch = torch.nn.utils.rnn.pack_padded_sequence(batch, lengths)
            _, (h, _) = self.lstm(batch)

            h_stepwise = hidden2fc_input(self.lstm, h)
        else:
            h_stepwise, (_, _) = self.lstm(x)

        x = h_stepwise
        for fc in self.fc:
            x = fc(x)

        # the flipping here comes not from the bidirectionality of the net,
        # but because the batch must be ordered longest first for PyTorch
        if self.lstm.bidirectional:
            x = torch.flip(x, dims=[0])

        return self.loss_delegate.mod_forward(x)

    def loss(self, data, lengths, orig_order, classes, training_config):
        output = self.forward((data, lengths))
        return self.loss_delegate(output, orig_order, classes, training_config), output


if __name__ == '__main__':
    sett = Settings('../configs/conf.ini', None)
    model = SiameseLSTM(None, sett)
    print(list(model.children()))
    print()
