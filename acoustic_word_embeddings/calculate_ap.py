import sys

import numpy as np

from acoustic_word_embeddings.core.util.args_util import parse_load_epoch_args
from acoustic_word_embeddings.core.average_precision import average_precision
from acoustic_word_embeddings.core.util.net_util import load_net
from acoustic_word_embeddings.gen_embeddings import get_siamese_embeddings, get_classifier_embeddings
from base.common import get_dataset_paths
from base.kaldi_dataset import KaldiDataset
from conf import current_dataset


def get_epoch_ap(net, config, checkpoints, loss, dataset, epoch, get_embeddings,
                 subsample_size=None):
    scores, classes, _ = get_embeddings(net, config, checkpoints[epoch], dataset)
    if subsample_size is not None:
        idx = np.random.choice(np.arange(scores.shape[0]), size=subsample_size, replace=False)
        scores = scores[idx]
        classes = classes[idx]
    return average_precision(scores, classes, metric=loss.metric() if loss is not None else 'cosine')


def do_calculate_ap(run_dir, epoch, dataset=None, partition='dev'):
    run_name2get_embeddings = {
        'classifier': get_classifier_embeddings,
        'siamese': get_siamese_embeddings
    }

    net, config, checkpoints, checkpoint_dir, run_name, loss, train_scp, _, _, _, mean_sub, var_norm = \
        load_net(run_dir, epoch=None, logger=None, train=False)
    get_embeddings = run_name2get_embeddings[run_name]

    if dataset is None:
        dataset = current_dataset

    train_path, dev_path, test_path = get_dataset_paths(dataset)

    if len(checkpoints) == 0:
        print('No checkpoints found in {0} for run {1}'.format(checkpoint_dir, run_dir))
        print('Exiting')
        sys.exit(-1)

    if partition == 'train':
        data_train = KaldiDataset('scp:' + train_path, parent_dataset_path=train_scp, training=False, logger=None,
                                  mean_subtraction=mean_sub, variance_normalization=var_norm)
        return get_epoch_ap(net, config, checkpoints, loss, data_train, epoch, get_embeddings,
                            subsample_size=3000)

    if partition == 'dev':
        data_dev = KaldiDataset('scp:' + dev_path, parent_dataset_path=train_scp, training=False, logger=None,
                                mean_subtraction=mean_sub, variance_normalization=var_norm)
        return get_epoch_ap(net, config, checkpoints, loss, data_dev, epoch, get_embeddings)

    if partition == 'test':
        data_test = KaldiDataset('scp:' + test_path, parent_dataset_path=train_scp, training=False, logger=None,
                                 mean_subtraction=mean_sub, variance_normalization=var_norm)
        return get_epoch_ap(net, config, checkpoints, loss, data_test, epoch, get_embeddings)


def _print_ap_per_epoch(net, config, checkpoints, loss, dataset, dataset_name, epochs_calc, get_embeddings,
                        subsample_size=None):
    for epoch in epochs_calc:
        scores, classes, _ = get_embeddings(net, config, checkpoints[epoch], dataset)
        if subsample_size is not None:
            idx = np.random.choice(np.arange(scores.shape[0]), size=subsample_size, replace=False)
            scores = scores[idx]
            classes = classes[idx]
        ap = average_precision(scores, classes, metric=loss.metric() if loss is not None else 'cosine',
                               shuffle_labels=False)
        print('Epoch {0}: {name} AP = {1:.3f}'.format(epoch, ap, name=dataset_name))


def __main(run_dir, dataset=None, for_epochs=None, gen_train=False, gen_dev=True, gen_test=False):
    run_name2get_embeddings = {
        'classifier': get_classifier_embeddings,
        'siamese': get_siamese_embeddings
    }

    net, config, checkpoints, checkpoint_dir, run_name, loss, train_scp, _, _, _, mean_sub, var_norm = \
        load_net(run_dir, epoch=None, logger=None, train=False)
    get_embeddings = run_name2get_embeddings[run_name]

    if dataset is None:
        dataset = current_dataset

    train_path, dev_path, test_path = get_dataset_paths(dataset)

    if len(checkpoints) == 0:
        print('No checkpoints found in {0} for run {1}'.format(checkpoint_dir, run_dir))
        print('Exiting')
        sys.exit(-1)

    if for_epochs is None:
        for_epochs = sorted(list(checkpoints.keys()))

    if gen_train:
        data_train = KaldiDataset('scp:' + train_path, parent_dataset_path=train_scp, training=False, logger=None,
                                  mean_subtraction=mean_sub, variance_normalization=var_norm)
        _print_ap_per_epoch(net, config, checkpoints, loss, data_train, 'train', for_epochs, get_embeddings,
                            subsample_size=3000)

    if gen_dev:
        data_dev = KaldiDataset('scp:' + dev_path, parent_dataset_path=train_scp, training=False, logger=None,
                                mean_subtraction=mean_sub, variance_normalization=var_norm)
        _print_ap_per_epoch(net, config, checkpoints, loss, data_dev, 'dev', for_epochs, get_embeddings)

    if gen_test:
        data_test = KaldiDataset('scp:' + test_path, parent_dataset_path=train_scp, training=False, logger=None,
                                 mean_subtraction=mean_sub, variance_normalization=var_norm)
        _print_ap_per_epoch(net, config, checkpoints, loss, data_test, 'test', for_epochs, get_embeddings)


if __name__ == '__main__':
    args = parse_load_epoch_args()
    __main(run_dir=args.run_dir, for_epochs=[args.run_epoch], gen_train=True, gen_dev=True, gen_test=False)
