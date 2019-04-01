import os
import time

import numpy as np
import torch

from acoustic_word_embeddings.core.net_util import setup_training_run
from acoustic_word_embeddings.nets.gru_classifier import GRUClassifier
from base.common import get_dataset_paths
from base.dataset import KaldiDataset
from acoustic_word_embeddings.nets.lstm_classifier import LSTMClassifier
from conf import current_dataset


def accuracy(correct_predictions_list):
    return np.sum(correct_predictions_list) / len(correct_predictions_list) * 100


def process_batch(data, lengths, sample_idx, orig_order, net, optimizer, dataset, config, train=True,
                  data_parallel=False):
    net.train(mode=train)

    classes_numpy = dataset.classes(sample_idx)
    data = torch.from_numpy(data)
    lengths = torch.from_numpy(lengths)
    classes = torch.from_numpy(classes_numpy).long()

    if config.model.use_cuda:
        data = data.cuda()
        lengths = lengths.cuda()
        classes = classes.cuda()

    if train:
        optimizer.zero_grad()

    if not data_parallel:
        loss, net_output = net.loss(data, lengths, orig_order, classes, config.classifier_training)
        predictions = net.predictions(net_output)
    else:
        loss, net_output = net.module.loss(data, lengths, orig_order, classes, config.classifier_training)
        predictions = net.module.predictions(net_output)

    if train:
        loss.backward()
        optimizer.step()
    return loss.item(), predictions, classes_numpy


def process_classifier_epoch(net, config, optimizer, dataset, batch_first, data_parallel, train):
    losses = []
    correct_predictions = []
    for data, lengths, sample_idx, orig_order in \
            dataset.classifier_epoch(batch_size=config.classifier_training.batch_size,
                                     batch_first=batch_first):
        loss, predictions, classes = process_batch(data, lengths, sample_idx, orig_order,
                                                   net, optimizer, dataset, config, train=train,
                                                   data_parallel=data_parallel)
        losses.append(loss)
        correct_predictions.extend(predictions == classes)
    return losses, accuracy(correct_predictions)


def __main():
    args, config, logger, checkpoint_dir, log_dir, use_gru, noise_mult, noise_prob, mean_sub, var_norm = \
        setup_training_run('classifier')

    supplement_rare = getattr(config.general_training, 'supplement_rare_with_noisy', False)
    supplement_seed = getattr(config.general_training, 'supplement_seed', 112)
    train_path, dev_path, _ = get_dataset_paths(current_dataset)
    data_train = KaldiDataset('scp:' + train_path, logger=logger, noise_multiplier=noise_mult, noise_prob=noise_prob,
                              mean_subtraction=mean_sub, variance_normalization=var_norm,
                              supplement_rare_with_noisy=supplement_rare, supplement_seed=supplement_seed)
    data_dev = KaldiDataset('scp:' + dev_path, parent_scp_path=train_path, training=False, logger=logger,
                            mean_subtraction=mean_sub, variance_normalization=var_norm)

    data_parallel = args.gpu_count > 1
    batch_first = data_parallel
    if not use_gru:
        net = LSTMClassifier(logger, config, batch_first=batch_first)
    else:
        net = GRUClassifier(logger, config, batch_first=batch_first)
    if args.load_weights is not None:
        net.restore_weights(args.load_weights, exclude_params=['fc.6'], freeze_except=None)
    net.train(True)

    if args.gpu_count > 1:
        net = torch.nn.DataParallel(net)
        config.classifier_training.batch_size = config.classifier_training.batch_size * args.gpu_count
        if config.model.use_cuda:
            net = net.cuda()

    optimizer = torch.optim.Adam(net.parameters(), lr=config.classifier_training.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.5, patience=7, min_lr=1e-5,
                                                           verbose=True)

    # log initial performance level
    dev_losses, dev_accuracy = process_classifier_epoch(net, config, optimizer, data_dev, batch_first, data_parallel, train=False)
    logger.info('Initial avg dev loss = {0:.4f}, dev accuracy = {1:.4f}'.format(np.mean(dev_losses), dev_accuracy))

    for epoch in range(config.classifier_training.train_epochs):
        logger.info('Starting epoch {0}, learning_rate = {1}'
                    .format(epoch, [group['lr'] for group in optimizer.param_groups][0]))

        start = time.time()
        train_losses, train_accuracy = process_classifier_epoch(net, config, optimizer, data_train, batch_first, data_parallel,
                                                                train=True)
        dev_losses, dev_accuracy = process_classifier_epoch(net, config, optimizer, data_dev, batch_first, data_parallel,
                                                            train=False)

        if config.classifier_training.lr_schedule:
            scheduler.step(dev_accuracy)

        torch.save(net.state_dict(),
                   os.path.join(checkpoint_dir, '{0}_epoch_{1}.ckpt'.format(net.__class__.__name__, epoch)))

        logger.info('Finished epoch {0}, avg training loss = {1:.4f}, avg dev loss = {2:.4f} epoch time = {3:.3f} sec'
                    .format(epoch, np.mean(train_losses), np.mean(dev_losses), time.time() - start))
        logger.info('Epoch {0} training accuracy = {1:.4f}, dev accuracy = {2:.4f}'
                    .format(epoch, train_accuracy, dev_accuracy))


if __name__ == '__main__':
    __main()
