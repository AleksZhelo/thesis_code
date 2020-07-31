import os
import pickle
import sys

import numpy as np
import torch

from acoustic_word_embeddings.core.util.args_util import parse_gen_args
from acoustic_word_embeddings.core.util.common import embeddings_dir2dict
from acoustic_word_embeddings.core.util.net_util import load_net
from base import util
from base.common import get_dataset_paths
from base.dataset import KaldiDataset
from acoustic_word_embeddings.nets.common import torch_load_unwrapped
from conf import current_dataset, new_path, processed_data_dir


def get_siamese_embeddings(net, config, checkpoint, dataset):
    embeddings = []
    classes = []
    data_idxs = []
    net.load_state_dict(torch_load_unwrapped(checkpoint))

    for data, lengths, data_idx, orig_idx in dataset.siamese_triplet_epoch(batch_size=768):
        data = torch.from_numpy(data)
        if config.model.use_cuda:
            data = data.cuda()

        net_output = net.forward((data, lengths))
        net_output = net_output.detach().cpu().numpy()

        embeddings.extend(net_output)
        classes.extend(dataset.classes(data_idx))
        data_idxs.extend(data_idx)

    return np.array(embeddings), np.array(classes), np.array(data_idxs)


def get_classifier_embeddings(net, config, checkpoint, dataset):
    embeddings = []
    classes = []
    data_idxs = []
    net.load_state_dict(torch_load_unwrapped(checkpoint))

    for data, lengths, data_idx, orig_idx in dataset.classifier_epoch(batch_size=768):
        data = torch.from_numpy(data)
        if config.model.use_cuda:
            data = data.cuda()

        net_input = torch.nn.utils.rnn.pack_padded_sequence(data, lengths)
        net_output = net.embedding(net_input)
        net_output = net_output.detach().cpu().numpy()

        embeddings.extend(net_output)
        classes.extend(dataset.classes(data_idx))
        data_idxs.extend(data_idx)

    return np.array(embeddings), np.array(classes), np.array(data_idxs)


def get_embeddings_dict(net, config, checkpoint, dataset, get_embeddings_fn):
    embedding_dict = {}

    embeddings, classes, data_idx = get_embeddings_fn(net, config, checkpoint, dataset)
    keys = dataset.idx2key[data_idx]

    for key, embedding in zip(keys, embeddings):
        embedding_dict[key] = embedding
    return embedding_dict


def save_embeddings(embedding_dict, embeddings_dir, checkpoint):
    epoch = util.checkpoint_path2epoch(checkpoint)
    output_path = os.path.join(embeddings_dir, 'embeddings_epoch_{0}.pickle'.format(epoch))
    with open(output_path, 'wb') as f:
        pickle.dump(embedding_dict, f, protocol=pickle.HIGHEST_PROTOCOL)


def gen_and_save_dataset_embeddings(net, config, checkpoint, dataset, get_embeddings, embeddings_dir):
    util.ensure_exists(embeddings_dir)
    embedding_dict = get_embeddings_dict(net, config, checkpoint, dataset, get_embeddings)
    save_embeddings(embedding_dict, embeddings_dir, checkpoint)


def generate_embeddings(run_dir, dataset=None, gen_train=False, gen_dev=False, gen_test=False, gen_new=False,
                        gen_background=False, for_epochs=None):
    run_name2emb = {
        'classifier': get_classifier_embeddings,
        'siamese': get_siamese_embeddings
    }

    net, config, checkpoints, checkpoint_dir, run_name, loss, train_scp, _, _, _, mean_sub, var_norm = \
        load_net(run_dir, epoch=None, logger=None, train=False)
    get_embeddings = run_name2emb[run_name]

    # XXX: currently if embeddings exists in train/dev/test folder they are not regenerated,
    #  remove manually when switching datasets
    if dataset is None:
        dataset = current_dataset
    train_path, dev_path, test_path = get_dataset_paths(dataset)

    if gen_train:
        data_train = KaldiDataset('scp:' + train_path, parent_scp_path=train_scp, training=False, logger=None,
                                  mean_subtraction=mean_sub, variance_normalization=var_norm)
    if gen_dev:
        data_dev = KaldiDataset('scp:' + dev_path, parent_scp_path=train_scp, training=False, logger=None,
                                mean_subtraction=mean_sub, variance_normalization=var_norm)
    if gen_test:
        data_test = KaldiDataset('scp:' + test_path, parent_scp_path=train_scp, training=False, logger=None,
                                 mean_subtraction=mean_sub, variance_normalization=var_norm)
    if gen_new:
        data_new = KaldiDataset('scp:' + new_path, parent_scp_path=train_scp, training=False, logger=None,
                                mean_subtraction=mean_sub, variance_normalization=var_norm)
    if gen_background:
        background_path = os.path.join(processed_data_dir, 'background_train_v4', 'background_data.scp')
        data_background = KaldiDataset('scp:' + background_path, parent_scp_path=train_scp, training=False, logger=None,
                                       mean_subtraction=mean_sub, variance_normalization=var_norm)

    train_embeddings_dir = os.path.join(run_dir, 'train_embeddings')
    dev_embeddings_dir = os.path.join(run_dir, 'dev_embeddings')
    test_embeddings_dir = os.path.join(run_dir, 'test_embeddings')
    new_embeddings_dir = os.path.join(run_dir, 'new_embeddings')
    background_embeddings_dir = os.path.join(run_dir, 'background_embeddings')

    if len(checkpoints) == 0:
        print('No checkpoints found in {0} for run {1}'.format(checkpoint_dir, run_dir))
        print('Exiting')
        sys.exit(-1)

    if for_epochs is None:
        for_epochs = sorted(list(checkpoints.keys()))

    for epoch in for_epochs:
        checkpoint = checkpoints[epoch]
        if gen_train:
            gen_and_save_dataset_embeddings(net, config, checkpoint, data_train, get_embeddings, train_embeddings_dir)

        if gen_dev:
            gen_and_save_dataset_embeddings(net, config, checkpoint, data_dev, get_embeddings, dev_embeddings_dir)

        if gen_test:
            gen_and_save_dataset_embeddings(net, config, checkpoint, data_test, get_embeddings, test_embeddings_dir)

        if gen_new:
            gen_and_save_dataset_embeddings(net, config, checkpoint, data_new, get_embeddings, new_embeddings_dir)

        if gen_background:
            gen_and_save_dataset_embeddings(net, config, checkpoint, data_background, get_embeddings,
                                            background_embeddings_dir)


def get_or_generate_embeddings(run_dir, epoch, dataset=None, dev_needed=False, test_needed=False):
    train_embeddings_dir = os.path.join(run_dir, 'train_embeddings')
    dev_embeddings_dir = os.path.join(run_dir, 'dev_embeddings')
    test_embeddings_dir = os.path.join(run_dir, 'test_embeddings')

    if not os.path.exists(train_embeddings_dir) or (not os.path.exists(dev_embeddings_dir) and dev_needed) \
            or (not os.path.exists(test_embeddings_dir) and test_needed):
        generate_embeddings(
            run_dir,
            dataset=dataset,
            gen_train=not os.path.exists(train_embeddings_dir),
            gen_dev=not os.path.exists(dev_embeddings_dir) and dev_needed,
            gen_test=not os.path.exists(test_embeddings_dir) and test_needed,
            for_epochs=[epoch]
        )

    train_epoch_embeddings = embeddings_dir2dict(train_embeddings_dir)
    dev_epoch_embeddings = embeddings_dir2dict(dev_embeddings_dir) if dev_needed else None
    test_epoch_embeddings = embeddings_dir2dict(test_embeddings_dir) if test_needed else None

    if epoch not in train_epoch_embeddings or (dev_needed and epoch not in dev_epoch_embeddings) or (
            test_needed and epoch not in test_epoch_embeddings):
        generate_embeddings(
            run_dir,
            dataset=dataset,
            gen_train=epoch not in train_epoch_embeddings,
            gen_dev=dev_needed and epoch not in dev_epoch_embeddings,
            gen_test=test_needed and epoch not in test_epoch_embeddings,
            for_epochs=[epoch]
        )
        train_epoch_embeddings = embeddings_dir2dict(train_embeddings_dir)
        dev_epoch_embeddings = embeddings_dir2dict(dev_embeddings_dir) if dev_needed else None
        test_epoch_embeddings = embeddings_dir2dict(test_embeddings_dir) if test_needed else None

    return train_epoch_embeddings, dev_epoch_embeddings, test_epoch_embeddings


if __name__ == '__main__':
    args = parse_gen_args()
    generate_embeddings(args.run_dir, gen_train=True, gen_dev=False, gen_test=False, gen_new=False,
                        gen_background=False, for_epochs=[17])
