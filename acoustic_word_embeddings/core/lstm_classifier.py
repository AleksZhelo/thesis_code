import torch
import torch.nn as nn

from base.settings import Settings
from acoustic_word_embeddings.nets.common import hidden2fc_input
from acoustic_word_embeddings.nets.lstm_fc_base import LSTM_FC_base


class LSTMClassifier(LSTM_FC_base):

    def __init__(self, logger, config, batch_first=False):
        super(LSTMClassifier, self).__init__(logger, config, batch_first)

    def _create_arch(self, settings):
        super(LSTMClassifier, self)._create_arch(settings)
        self.fc.append(nn.LogSoftmax(dim=1))
        self.nll_loss = nn.CrossEntropyLoss()

    def embedding(self, x):
        _, (h, _) = self.lstm(x)  # hidden state defaults to zero

        x = hidden2fc_input(self.lstm, h)
        for fc in self.fc[:-1]:
            x = fc(x)

        return x

    def loss(self, data, lengths, orig_order, classes, training_config):
        output = self.forward((data, lengths))
        return self.nll_loss(output, classes), output

    def predictions(self, net_output):
        return torch.argmax(net_output, dim=1).detach().cpu().numpy()

    def accuracy(self, net_output, classes):
        return torch.sum(torch.argmax(net_output, dim=1) == classes).item() / classes.size()[0] * 100


if __name__ == '__main__':
    sett = Settings('configs/conf.ini', None)
    model = LSTMClassifier(None, sett)
    print(model.children())
    print()
