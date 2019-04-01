from abc import ABC

import torch

from base.settings import Settings
from base.torch.common import hidden2fc_input
from base.torch.experimental.custom_gru import CustomGRU, BiRNNMode
from base.torch.model import Model
import torch.nn as nn


# noinspection PyPep8Naming
class GRU_FC_base(Model, ABC):

    def __init__(self, logger, config, batch_first=False):
        super(GRU_FC_base, self).__init__(logger, config, batch_first)

        self.init_type = None
        self.init_param = None

        self._create_arch(config)
        self._init()

    def _create_arch(self, settings):
        gru_fc_base_conf = getattr(settings, GRU_FC_base.__name__)
        dropout = self.my_settings.dropout
        fc_dropout = self.my_settings.fc_dropout if hasattr(self.my_settings, 'fc_dropout') else []
        self.init_type = gru_fc_base_conf.init_type.strip() if hasattr(gru_fc_base_conf,
                                                                       'init_type') else 'xavier_normal'
        self.init_param = gru_fc_base_conf.init_param if hasattr(gru_fc_base_conf, 'init_param') else 1

        if not hasattr(gru_fc_base_conf, 'use_custom') or not gru_fc_base_conf.use_custom:
            self.gru = nn.GRU(self.input_size, gru_fc_base_conf.gru_hidden_size, gru_fc_base_conf.gru_layers,
                              bias=True, dropout=dropout, bidirectional=gru_fc_base_conf.bidirectional,
                              batch_first=self.batch_first)
        else:
            dropout_last = gru_fc_base_conf.custom_dropout_last
            dropout_hidden = gru_fc_base_conf.custom_dropout_hidden
            reverse_backward_output = hasattr(gru_fc_base_conf, 'custom_reverse_backward_output') \
                                      and gru_fc_base_conf.custom_reverse_backward_output
            mode = gru_fc_base_conf.custom_mode if hasattr(gru_fc_base_conf, 'custom_mode') else 'CONCAT'
            mode = BiRNNMode[mode]
            decov_output = gru_fc_base_conf.custom_decov_output if hasattr(gru_fc_base_conf,
                                                                           'custom_decov_output') else []
            decov_hidden = gru_fc_base_conf.custom_decov_hidden if hasattr(gru_fc_base_conf,
                                                                           'custom_decov_hidden') else []
            self.gru = CustomGRU(self.input_size, gru_fc_base_conf.gru_hidden_size, gru_fc_base_conf.gru_layers,
                                 bias=True, dropout=dropout, bidirectional=gru_fc_base_conf.bidirectional,
                                 batch_first=self.batch_first, dropout_last=dropout_last, dropout_hidden=dropout_hidden,
                                 mode=mode, reverse_backward_output=reverse_backward_output, decov_output=decov_output,
                                 decov_hidden=decov_hidden)

        self.fc = nn.ModuleList()
        input_size = gru_fc_base_conf.gru_hidden_size
        if gru_fc_base_conf.bidirectional:
            input_size *= 2
        for fc_idx, h in enumerate(gru_fc_base_conf.fc_size):
            self.fc.append(nn.Linear(input_size, h))
            self.fc.append(nn.ReLU())
            if fc_idx < len(fc_dropout) and fc_dropout[fc_idx] > 0:
                self.fc.append(torch.nn.Dropout(p=fc_dropout[fc_idx]))
            input_size = h
        self.fc.append(nn.Linear(input_size, self.output_size))

    def _init_weights(self):
        for name, param in self.gru.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                if self.init_type == 'xavier_normal':
                    nn.init.xavier_normal_(param, gain=self.init_param)
                elif self.init_type == 'xavier_uniform':
                    nn.init.xavier_uniform_(param, gain=self.init_param)
                elif self.init_type == 'uniform':
                    nn.init.uniform_(param, a=-self.init_param, b=self.init_param)
                else:
                    raise RuntimeError('Unknown GRU weight init type: {0}'.format(self.init_type))

        for name, param in self.fc.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param,
                                       gain=nn.init.calculate_gain('relu') if str(len(self.fc) - 1) not in name else 1)

    def forward(self, x):
        data, lengths = x
        net_input = torch.nn.utils.rnn.pack_padded_sequence(data, lengths, batch_first=self.batch_first)
        _, h = self.gru(net_input)  # hidden state defaults to zero

        x = hidden2fc_input(self.gru, h)
        for fc in self.fc:
            x = fc(x)

        return x


if __name__ == '__main__':
    sett = Settings('configs/conf.ini', None)
    model = GRU_FC_base(None, sett)
    print(model.children())
    print()
