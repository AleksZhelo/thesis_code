import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from acoustic_word_embeddings.core.util.common import load_embeddings
from acoustic_word_embeddings.core.loss.embedding_loss import loss_name2class
from acoustic_word_embeddings.core.util.net_util import read_embedding_loss, load_net
from acoustic_word_embeddings.gen_embeddings import get_or_generate_embeddings
from acoustic_word_embeddings.train_classifier import process_classifier_epoch
from base.common import get_dataset_paths
from base.dataset import KaldiDataset
from conf import current_dataset


def do_calculate_accuracy(run_dir, epoch, is_classifier, dataset=None, partition='dev', return_percent=False):
    if not is_classifier:
        train_epoch_embeddings, dev_epoch_embeddings, test_epoch_embeddings = \
            get_or_generate_embeddings(run_dir, epoch, dataset=dataset,
                                       dev_needed=(partition == 'dev'), test_needed=(partition == 'test'))

        words_train, datasets_train, vecs_train, counts_train, word_idxs_train = load_embeddings(
            train_epoch_embeddings[epoch], data_name='train')

        if partition == 'dev':
            words_part, datasets_part, vecs_part, counts_part, word_idxs_part = load_embeddings(
                dev_epoch_embeddings[epoch])
        elif partition == 'test':
            words_part, datasets_part, vecs_part, counts_part, word_idxs_part = load_embeddings(
                test_epoch_embeddings[epoch],
                data_name='test'
            )
        else:
            raise RuntimeError('Cannot calculate accuracy for partition {0}'.format(partition))

        test_word_idxs = word_idxs_part
        test_vecs = vecs_part

        all_words = set()
        all_words.update(word_idxs_train.keys())
        all_words.update(test_word_idxs.keys())

        max_reference_examples = 40
        train_x = []
        train_y = []
        test_x = []
        test_y = []
        for i, word in enumerate(all_words):
            if word in word_idxs_train:
                all_word_idxs = word_idxs_train[word]
                n_train = min(int(np.floor(all_word_idxs.shape[0] / 2)), max_reference_examples)
                train_x.extend(vecs_train[all_word_idxs[:n_train]])
                train_y.extend([i] * n_train)

                if word_idxs_train != test_word_idxs:
                    if word in test_word_idxs:
                        test_dat = test_vecs[test_word_idxs[word]]
                        test_x.extend(test_dat)
                        test_y.extend([i] * test_dat.shape[0])
                else:
                    test_dat = vecs_train[all_word_idxs[n_train:]]
                    test_x.extend(test_dat)
                    test_y.extend([i] * test_dat.shape[0])

            elif word in test_word_idxs:
                all_word_idxs = test_word_idxs[word]
                n_train = min(int(np.floor(all_word_idxs.shape[0] / 2)), max_reference_examples)
                train_x.extend(test_vecs[all_word_idxs[:n_train]])
                train_y.extend([i] * n_train)

                test_dat = test_vecs[all_word_idxs[n_train:]]
                test_x.extend(test_dat)
                test_y.extend([i] * test_dat.shape[0])

        loss_name = read_embedding_loss(run_dir, throw=False)
        metric = loss_name2class[loss_name].metric(None) if loss_name is not None else 'cosine'

        print(run_dir, metric)
        print('N train: {0}, N test: {1}'.format(len(train_x), len(test_x)))

        knn = KNeighborsClassifier(n_neighbors=3, metric=metric, n_jobs=8)
        knn.fit(train_x, train_y)
        k_pred_y = knn.predict(test_x)

        if not return_percent:
            return np.sum(k_pred_y == test_y) / len(test_y)
        else:
            return np.sum(k_pred_y == test_y) / len(test_y) * 100.0
    else:
        net, config, checkpoints, checkpoint_dir, run_name, loss, train_scp, _, _, _, mean_sub, var_norm = \
            load_net(run_dir, epoch=epoch, logger=None, train=False)

        if dataset is None:
            dataset = current_dataset
        train_path, dev_path, test_path = get_dataset_paths(dataset)

        if partition == 'train':
            dataset = KaldiDataset('scp:' + train_path, parent_scp_path=train_scp, training=False, logger=None,
                                   mean_subtraction=mean_sub, variance_normalization=var_norm)
        if partition == 'dev':
            dataset = KaldiDataset('scp:' + dev_path, parent_scp_path=train_scp, training=False, logger=None,
                                   mean_subtraction=mean_sub, variance_normalization=var_norm)
        if partition == 'test':
            dataset = KaldiDataset('scp:' + test_path, parent_scp_path=train_scp, training=False, logger=None,
                                   mean_subtraction=mean_sub, variance_normalization=var_norm)

        # TODO: no automatic detection for batch_first and data_parallel
        losses, accuracy = process_classifier_epoch(net, config, optimizer=None, dataset=dataset, batch_first=False,
                                                    data_parallel=False, train=False)
        if not return_percent:
            return accuracy / 100.0
        else:
            return accuracy
