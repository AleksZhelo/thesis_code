import numpy as np
import torch


def _distance_matrix(x):
    square = x.pow(2).sum(dim=1, keepdim=True)
    distance_square = square - 2.0 * torch.mm(x, x.t()) + square.t()

    # Adding identity to make sqrt work
    return torch.sqrt(distance_square + torch.eye(x.shape[0], dtype=x.dtype, device=x.device))


def sample_distance_weighted(net_output, orig_order, training_config, cutoff, nonzero_loss_cutoff):
    """Source: https://github.com/chaoyuaw/incubator-mxnet/blob/master/example/gluon/embedding_learning/model.py"""

    net_output = net_output[orig_order]
    n = training_config.online_batch_size
    k = training_config.online_examples_per_word
    dim = net_output.size()[1]

    distance = _distance_matrix(net_output)
    distance = torch.clamp(distance, min=cutoff)

    log_weights = (2.0 - dim) * torch.log(distance) - ((dim - 3) / 2) * torch.log(1.0 - 0.25 * distance.pow(2))
    weights = torch.exp(log_weights - log_weights.max())  # Subtracting maximum for stability

    # Sample only negative examples by setting weights of the same-class examples to 0
    mask = np.ones(weights.shape, dtype=np.float32)
    for i in range(0, n, k):
        mask[i:i + k, i:i + k] = 0

    mask = torch.from_numpy(mask)
    if weights.is_cuda:
        mask = mask.cuda()

    weights = (weights * mask) * (distance < nonzero_loss_cutoff).type(torch.float32)
    weights = weights / weights.sum(dim=1, keepdim=True)

    anchor_idx = []
    same_idx = []
    other_idx = []

    np_weights = weights.detach().cpu().numpy()
    for i in range(n):
        block_idx = i // k

        try:
            other_idx += np.random.choice(n, k - 1, p=np_weights[i]).tolist()
        except:
            other_idx += np.random.choice(n, k - 1).tolist()
        for j in range(block_idx * k, (block_idx + 1) * k):
            if j != i:
                anchor_idx.append(i)
                same_idx.append(j)

    return net_output[anchor_idx], net_output[same_idx], net_output[other_idx], anchor_idx
