from abc import ABC

import torch
import torch.nn as nn

from base.settings import Settings
from base.torch.common import hidden2fc_input
from base.torch.model import Model


# noinspection PyPep8Naming
class LSTM_FC_base(Model, ABC):

    def __init__(self, logger, config, batch_first=False):
        super(LSTM_FC_base, self).__init__(logger, config, batch_first)

        self.init_type = None
        self.init_param = None

        self._create_arch(config)
        self._init()

    def _create_arch(self, settings):
        lstm_fc_base_conf = getattr(settings, LSTM_FC_base.__name__)
        dropout = self.my_settings.dropout
        fc_dropout = self.my_settings.fc_dropout if hasattr(self.my_settings, 'fc_dropout') else []
        self.init_type = lstm_fc_base_conf.init_type.strip() if hasattr(lstm_fc_base_conf,
                                                                        'init_type') else 'xavier_normal'
        self.init_param = lstm_fc_base_conf.init_param if hasattr(lstm_fc_base_conf, 'init_param') else 1

        self.lstm = nn.LSTM(self.input_size, lstm_fc_base_conf.lstm_hidden_size, lstm_fc_base_conf.lstm_layers,
                            bias=True, dropout=dropout, bidirectional=lstm_fc_base_conf.bidirectional,
                            batch_first=self.batch_first)

        self.fc = nn.ModuleList()
        input_size = lstm_fc_base_conf.lstm_hidden_size
        if lstm_fc_base_conf.bidirectional:
            input_size *= 2
        for fc_idx, h in enumerate(lstm_fc_base_conf.fc_size):
            self.fc.append(nn.Linear(input_size, h))
            self.fc.append(nn.ReLU())
            if fc_idx < len(fc_dropout) and fc_dropout[fc_idx] > 0:
                self.fc.append(torch.nn.Dropout(p=fc_dropout[fc_idx]))
            input_size = h
        self.fc.append(nn.Linear(input_size, self.output_size))

    def _init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                # nn.init.uniform_(param, 0.1, 2)
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                if self.init_type == 'xavier_normal':
                    nn.init.xavier_normal_(param, gain=self.init_param)
                elif self.init_type == 'xavier_uniform':
                    nn.init.xavier_uniform_(param, gain=self.init_param)
                elif self.init_type == 'uniform':
                    nn.init.uniform_(param, a=-self.init_param, b=self.init_param)
                else:
                    raise RuntimeError('Unknown LSTM weight init type: {0}'.format(self.init_type))

        for name, param in self.fc.named_parameters():
            if 'bias' in name:
                # nn.init.uniform_(param, 0.1, 2)
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param,
                                       gain=nn.init.calculate_gain('relu') if str(len(self.fc) - 1) not in name else 1)

    def forward(self, x):
        data, lengths = x
        net_input = torch.nn.utils.rnn.pack_padded_sequence(data, lengths, batch_first=self.batch_first)
        _, (h, _) = self.lstm(net_input)  # hidden state defaults to zero

        x = hidden2fc_input(self.lstm, h)
        for fc in self.fc:
            x = fc(x)

        return x


if __name__ == '__main__':
    sett = Settings('configs/conf.ini', None)
    model = LSTM_FC_base(None, sett)
    print(model.children())
    print()
