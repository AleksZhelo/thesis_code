from collections.__init__ import OrderedDict

import torch


def hidden2fc_input(rnn, h):
    if rnn.bidirectional:
        h = h.view(rnn.num_layers, 2, h.size()[1], rnn.hidden_size)
        x = torch.cat((h[-1, 0], h[-1, 1]), dim=1)
    else:
        x = h[-1]
    return x


def torch_load_unwrapped(path):
    """Fixing the changes induced by wrapping with DataParallel"""
    state_dict = torch.load(path)
    fixed_dict = OrderedDict()
    for key in state_dict:
        if key.startswith('module.'):
            fixed_dict[key[7:]] = state_dict[key]
        else:
            fixed_dict[key] = state_dict[key]
    return fixed_dict


def get_dropout_mask(dropout_probability: float, tensor_for_masking: torch.Tensor, training: bool):
    """
    Adapted from https://github.com/allenai/allennlp/blob/master/allennlp/nn/util.py
    """
    if dropout_probability > 0 and training:
        binary_mask = tensor_for_masking.new_tensor(torch.rand(tensor_for_masking.size()) > dropout_probability,
                                                    requires_grad=False)
        # Scale mask by 1/keep_prob to preserve output statistics.
        dropout_mask = binary_mask.float().div(1.0 - dropout_probability)
        return dropout_mask
    else:
        return None


def get_cuda_tensors():
    import gc
    for obj in gc.get_objects():
        # noinspection PyBroadException
        try:
            if torch.is_tensor(obj):
                tensor = obj
            elif hasattr(obj, 'data') and torch.is_tensor(obj.data):
                tensor = obj.data
            else:
                continue

            if tensor.is_cuda:
                yield tensor
        except Exception:
            pass
