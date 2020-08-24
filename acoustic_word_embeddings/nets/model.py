import torch

from acoustic_word_embeddings.nets.common import torch_load_unwrapped
from base import util


class Model(torch.nn.Module):
    def __init__(self, logger, config, batch_first=False):
        super(Model, self).__init__()

        self.logger = logger
        self.use_cuda = config.model.use_cuda
        self.dtype = torch.cuda.FloatTensor if self.use_cuda else torch.FloatTensor
        self.batch_first = batch_first

        self.my_settings = getattr(config, self.__class__.__name__)
        self.input_size = config.model.input_size
        self.output_size = self.my_settings.output_size

    def _init_weights(self):
        raise NotImplementedError()

    def print_model(self):
        if self.logger is not None:
            self.logger.warning('----------- {0} -----------'.format(self.__class__.__name__))
            self.logger.warning(self)

    def _init(self):
        self._init_weights()
        self.type(self.dtype)
        self.print_model()

    def forward(self, x):
        """
        :param x: input sequences as (data, lengths)
        :return: model output for each input sequence
        """
        raise NotImplementedError()

    def loss(self, data, lengths, orig_order, classes, training_config):
        raise NotImplementedError()

    def restore_weights(self, path, exclude_params, freeze_except=None, strict=True):
        weights = torch_load_unwrapped(path)

        for name, param in self.named_parameters():
            if name not in weights:
                if not strict:
                    continue
                else:
                    msg = '{0} not in the given checkpoint file: {1}'.format(name, path)
                    util.critical_or_print(self.logger, msg)
                    raise Exception(msg)

            if all([x not in name for x in exclude_params]):
                if param.data.size() != weights[name].size():
                    msg = '{0} size is not correct: should be {1}, checkpoint data has {2}' \
                        .format(name, param.data.size(), weights[name].size())
                    util.critical_or_print(self.logger, msg)
                    raise Exception(msg)
                param.data = weights[name].data
                if freeze_except is not None:
                    param.required_grad = any([x in name for x in freeze_except])

        if self.logger is not None:
            self.logger.info('Restored from {0}, excluded {1}'.format(path, exclude_params))
            if freeze_except is not None:
                self.logger.info('Freezing all loaded except {0})'.format(freeze_except))

    def parameter_count(self):
        """https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/9"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
