import torch

from acoustic_word_embeddings.core.sampling import sample_distance_weighted


# TODO: should have been decoupled from the sampling process, removing distance_weighted_sampling param and leaving that
#  to the model composition stage
class DistanceBasedLoss(object):

    # noinspection PyUnusedLocal
    def __init__(self, config, num_classes, cuda=False):
        pass

    def __call__(self, net_output, orig_order, classes, training_config):
        raise NotImplementedError()

    def register_params(self, net):
        pass

    def mod_forward(self, x):
        return x

    def metric(self):
        return 'cosine'

    def distance_weighted_sampling(self):
        return False

    @staticmethod
    def create(name, config, num_classes):
        return loss_name2class[name](config, num_classes, cuda=config.model.use_cuda)


class EuclideanLoss(DistanceBasedLoss):

    def __call__(self, net_output, orig_order, classes, training_config):
        raise NotImplementedError()

    def mod_forward(self, x):
        return torch.nn.functional.normalize(x)

    def metric(self):
        return 'euclidean'


class triplet_loss_offline(DistanceBasedLoss):
    """Following DISCRIMINATIVE ACOUSTIC WORD EMBEDDINGS: RECURRENT NEURAL NETWORK-BASED APPROACHES"""
    sim = torch.nn.CosineSimilarity(dim=1)

    def __call__(self, net_output, orig_order, classes, training_config):
        anchor, same, *diff = net_output[orig_order].split(int(len(orig_order) / (2 + training_config.num_other)))

        max_sim_diff = self.sim(anchor.unsqueeze(2).expand(-1, -1, len(diff)), torch.stack(diff, 2)).max(1)[0]
        sim_same = self.sim(anchor, same)
        loss = torch.clamp(training_config.loss_margin - sim_same + max_sim_diff, min=0)
        return torch.mean(loss)


class triplet_loss_offline_euclidean(EuclideanLoss):

    def __call__(self, net_output, orig_order, classes, training_config):
        anchor, same, *diff = net_output[orig_order].split(int(len(orig_order) / (2 + training_config.num_other)))

        min_dist_diff = torch.norm(anchor.unsqueeze(2).expand(-1, -1, len(diff)) - torch.stack(diff, 2), p=2, dim=1) \
            .min(1)[0]
        dist_same = torch.norm(anchor - same, p=2, dim=1)
        loss = torch.clamp(training_config.euclidean_loss_margin + dist_same - min_dist_diff, min=0)
        return torch.mean(loss)


class triplet_loss_offline_squared_euclidean(EuclideanLoss):
    """Following FaceNet: A Unified Embedding for Face Recognition and Clustering

    As the embeddings are L2-normalized, the squared euclidean distance in proportional to the cosine distance:
    ||x - y||^2 = 2 - 2*x^T*y = 2 - 2*cos_sim(x, y)

    """

    def metric(self):
        return 'sqeuclidean'

    # Squared euclidean seems to be performing poorly
    # Actually it was due to me messing up the calculations: I should be penalizing minimum distance to other,
    # not maximum! Fixed now
    def __call__(self, net_output, orig_order, classes, training_config):
        anchor, same, *diff = net_output[orig_order].split(int(len(orig_order) / (2 + training_config.num_other)))

        # L2-norm squared
        min_dist_diff = (anchor.unsqueeze(2).expand(-1, -1, len(diff)) - torch.stack(diff, 2)).pow(2).sum(1).min(1)[0]
        dist_same = (anchor - same).pow(2).sum(1)
        loss = torch.clamp(training_config.euclidean_loss_margin + dist_same - min_dist_diff, min=0)
        return torch.mean(loss)


class triplet_loss_online_euclidean(EuclideanLoss):
    """Following https://arxiv.org/pdf/1706.07567.pdf - Sampling Matters in Deep Embedding Learning
    """

    def __init__(self, config, num_classes, cuda=False):
        super(triplet_loss_online_euclidean, self).__init__(config, num_classes, cuda)
        self.cutoff = config.siamese_training.margin_cutoff
        self.nonzero_loss_cutoff = config.siamese_training.margin_nonzero_loss_cutoff

    def distance_weighted_sampling(self):
        return True

    def __call__(self, net_output, orig_order, classes, training_config):
        anchor, same, other, _ = sample_distance_weighted(net_output, orig_order, training_config, self.cutoff,
                                                          self.nonzero_loss_cutoff)

        dist_diff = torch.norm(anchor - other, p=2, dim=1)
        dist_same = torch.norm(anchor - same, p=2, dim=1)
        loss = torch.clamp(training_config.euclidean_loss_margin + dist_same - dist_diff, min=0)
        return torch.mean(loss)


class margin_loss(EuclideanLoss):
    """Following https://arxiv.org/pdf/1706.07567.pdf - Sampling Matters in Deep Embedding Learning

    Source: https://github.com/chaoyuaw/incubator-mxnet/blob/master/example/gluon/embedding_learning/model.py
    """

    def __init__(self, config, num_classes, cuda=False):
        super(margin_loss, self).__init__(config, num_classes, cuda)
        self.cutoff = config.siamese_training.margin_cutoff
        self.nonzero_loss_cutoff = config.siamese_training.margin_nonzero_loss_cutoff
        self.init_beta = config.siamese_training.margin_init_beta
        self.nu = config.siamese_training.margin_nu

        beta = torch.empty(num_classes, requires_grad=True)
        if cuda:
            beta = beta.cuda()
        torch.nn.init.constant_(beta, self.init_beta)
        self.beta = torch.nn.Parameter(beta)

    def register_params(self, net):
        super(margin_loss, self).register_params(net)
        net.register_parameter('beta', self.beta)

    def distance_weighted_sampling(self):
        return True

    def __call__(self, net_output, orig_order, classes, training_config):
        anchor, same, other, anchor_idx = sample_distance_weighted(net_output, orig_order, training_config, self.cutoff,
                                                                   self.nonzero_loss_cutoff)
        beta = self.beta[classes[orig_order][anchor_idx]]
        beta_reg_loss = beta.sum() * self.nu

        dist_diff = torch.norm(anchor - other, p=2, dim=1)
        dist_same = torch.norm(anchor - same, p=2, dim=1)

        same_loss = torch.clamp(dist_same - beta + training_config.euclidean_loss_margin, min=0)
        diff_loss = torch.clamp(beta - dist_diff + training_config.euclidean_loss_margin, min=0)

        pair_count = ((same_loss > 0.0) + (diff_loss > 0.0)).sum().item()
        return ((same_loss + diff_loss).sum() + beta_reg_loss) / pair_count


loss_name2class = {
    'triplet_loss_offline': triplet_loss_offline,
    'triplet_loss_offline_euclidean': triplet_loss_offline_euclidean,
    'triplet_loss_offline_squared_euclidean': triplet_loss_offline_squared_euclidean,
    'triplet_loss_online_euclidean': triplet_loss_online_euclidean,
    'margin_loss': margin_loss
}
