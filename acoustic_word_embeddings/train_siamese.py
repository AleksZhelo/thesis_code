import gc
import os
import time

import numpy as np
import torch

from acoustic_word_embeddings.core.average_precision import average_precision
from acoustic_word_embeddings.core.loss.embedding_loss import margin_loss
from acoustic_word_embeddings.core.util.net_util import setup_training_run, create_embedding_loss
from acoustic_word_embeddings.core.siamese_gru import SiameseGRU
from acoustic_word_embeddings.core.siamese_lstm import SiameseLSTM
from base.common import get_dataset_paths
from base.data_io.kaldi_dataset import KaldiDataset
from conf import current_dataset


def create_optimizer(net, config, loss, wrapped=False):
    if isinstance(loss, margin_loss):
        beta_name = 'beta' if not wrapped else 'module.beta'
        param_list = [
            {'params': [val for name, val in net.named_parameters() if name != beta_name]},
            {'params': [dict(net.named_parameters())[beta_name]], 'lr': config.siamese_training.lr_margin_beta}
        ]
        optimizer = torch.optim.Adam(param_list, lr=config.siamese_training.learning_rate)
    else:
        optimizer = torch.optim.Adam(net.parameters(), lr=config.siamese_training.learning_rate)
    return optimizer


def init_batch_generator(config, dataset, loss_fn, augment_parts, data_parallel):
    if loss_fn.distance_weighted_sampling():
        return dataset.siamese_margin_loss_epoch(
            batch_size=config.siamese_training.online_batch_size,
            examples_per_word=config.siamese_training.online_examples_per_word,
            augment_parts=augment_parts, batch_first=data_parallel
        )
    else:
        return dataset.siamese_triplet_epoch(
            batch_size=config.siamese_training.batch_size,
            num_other=config.siamese_training.num_other,
            augment_parts=augment_parts, batch_first=data_parallel
        )


def process_siamese_batch(data, lengths, classes, orig_order, net, optimizer, config, train=True, data_parallel=False):
    net.train(mode=train)

    data = torch.from_numpy(data)
    lengths = torch.from_numpy(lengths)
    classes = torch.from_numpy(classes).long() if classes is not None else None
    if config.model.use_cuda:
        data = data.cuda()
        lengths = lengths.cuda()
        classes = classes.cuda() if classes is not None else None

    if train:
        optimizer.zero_grad()
        if not data_parallel:
            loss, net_output = net.loss(data, lengths, orig_order, classes, config.siamese_training)
        else:
            loss, net_output = net.module.loss(data, lengths, orig_order, classes, config.siamese_training)
        out_loss = loss.item()
    else:
        net_output = net.forward((data, lengths))
        out_loss = None

    if train:
        loss.backward()
        optimizer.step()
    return out_loss, net_output


def get_ap(net, metric, optimizer, config, dataset, data_parallel=False):
    batch_embeddings = []
    batch_classes = []
    for data, lengths, sample_idx, orig_order in \
            dataset.siamese_triplet_epoch(batch_size=config.siamese_training.batch_size * 10,
                                          batch_first=data_parallel):
        loss, embeddings = process_siamese_batch(data, lengths, dataset.classes(sample_idx), orig_order, net, optimizer,
                                                 config,
                                                 train=False, data_parallel=data_parallel)

        batch_embeddings.extend(embeddings.detach().cpu().numpy())
        batch_classes.extend(dataset.classes(sample_idx))
    ap = average_precision(np.array(batch_embeddings), np.array(batch_classes), metric=metric)
    return ap


def __main():
    args, config, logger, checkpoint_dir, log_dir, use_gru, noise_mult, noise_prob, mean_sub, var_norm = \
        setup_training_run('siamese')

    supplement_rare = getattr(config.general_training, 'supplement_rare_with_noisy', False)
    supplement_seed = getattr(config.general_training, 'supplement_seed', 112)
    train_path, dev_path, _ = get_dataset_paths(current_dataset)
    data_train = KaldiDataset('scp:' + train_path, logger=logger, noise_multiplier=noise_mult, noise_prob=noise_prob,
                              mean_subtraction=mean_sub, variance_normalization=var_norm,
                              supplement_rare_with_noisy=supplement_rare, supplement_seed=supplement_seed)
    data_dev = KaldiDataset('scp:' + dev_path, parent_dataset_path=train_path, training=False, logger=logger,
                            mean_subtraction=mean_sub, variance_normalization=var_norm)

    loss_fn = create_embedding_loss(config, len(data_train.word2id))
    data_parallel = args.gpu_count > 1
    batch_first = data_parallel

    if not use_gru:
        net = SiameseLSTM(logger, config, batch_first=batch_first, loss=loss_fn)
    else:
        net = SiameseGRU(logger, config, batch_first=batch_first, loss=loss_fn)

    if args.load_weights is not None:
        # exclude_params specifies the layers to drop when applying pre-trained weights
        net.restore_weights(args.load_weights, exclude_params=['fc.2'], freeze_except=None)
        # net.restore_weights(args.load_weights, exclude_params=[], freeze_except=None)
    net.train(True)

    if args.gpu_count > 1:
        net = torch.nn.DataParallel(net)
        config.siamese_training.batch_size = config.siamese_training.batch_size * args.gpu_count
        if config.model.use_cuda:
            net = net.cuda()

    optimizer = create_optimizer(net, config, loss_fn, wrapped=data_parallel)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.5, patience=6, min_lr=1e-5,
                                                           verbose=True)

    # other settings
    augment_parts = config.siamese_training.augment_parts if hasattr(config.siamese_training,
                                                                     'augment_parts') else False

    # log initial performance level
    dev_ap = get_ap(net, loss_fn.metric(), optimizer, config, data_dev, batch_first)
    logger.info('Initial avg dev precision= {0:.4f}'.format(dev_ap))

    for epoch in range(config.siamese_training.train_epochs):
        logger.info('Starting epoch {0}, learning_rate = {1}'
                    .format(epoch, [group['lr'] for group in optimizer.param_groups]))

        start = time.time()
        epoch_losses = []
        for data, lengths, sample_idx, orig_order in init_batch_generator(config, data_train, loss_fn, augment_parts,
                                                                          batch_first):
            loss, _ = process_siamese_batch(data, lengths, data_train.classes(sample_idx), orig_order, net, optimizer,
                                            config,
                                            train=True, data_parallel=data_parallel)
            del _
            gc.collect()
            epoch_losses.append(loss)

        dev_ap = get_ap(net, loss_fn.metric(), optimizer, config, data_dev, batch_first)

        if config.siamese_training.lr_schedule:
            scheduler.step(dev_ap)

        torch.save(net.state_dict(),
                   os.path.join(checkpoint_dir, '{0}_epoch_{1}.ckpt'.format(net.__class__.__name__, epoch)))

        logger.info('Finished epoch {0}, average training loss = {1:.4f}, epoch time = {2:.3f} sec'
                    .format(epoch, np.mean(epoch_losses), time.time() - start))
        logger.info('Epoch {0} avg dev precision= {1:.4f}'.format(epoch, dev_ap))


if __name__ == '__main__':
    __main()
