import torch


def decov_3d(activations_tensor, batch_lengths):
    lens_float = batch_lengths.float().to(activations_tensor.device)
    cov = 0
    for i, length in enumerate(batch_lengths):
        sequence = activations_tensor[:length, i, :]
        v = sequence - sequence.mean(dim=0)
        cov += torch.mm(v.transpose(0, 1), v)
    cov = cov / (torch.sum(lens_float) - lens_float.size()[0])
    return 0.5 * (cov.pow(2).sum() - torch.diag(cov).pow(2).sum())


def decov_2d(activations_tensor):
    v = activations_tensor - activations_tensor.mean(dim=0)
    cov = torch.mm(v.transpose(0, 1), v) / (activations_tensor.size()[0] - 1)
    return 0.5 * (cov.pow(2).sum() - torch.diag(cov).pow(2).sum())
