import os
import time
from collections import Counter

import kaldi_io
import numpy as np

from base.common import get_dataset_paths, snodgrass_key2patient, snodgrass_key2date, key2word
from base.data_io.dataset import Dataset, _print_patients
from base.sound_util import frames2time
from conf import current_dataset, processed_data_dir


class KaldiDataset(Dataset):
    def __init__(self, data_path, parent_dataset_path=None, training=True, logger=None, variance_normalization=False,
                 noise_multiplier=0, noise_prob=1, mean_subtraction=False, supplement_rare_with_noisy=False,
                 supplement_seed=112):
        super(KaldiDataset, self).__init__(data_path, parent_dataset_path, training, logger, variance_normalization,
                                           noise_multiplier, noise_prob, mean_subtraction, supplement_rare_with_noisy,
                                           supplement_seed)

    def _raw_data_iterator(self):
        return kaldi_io.read_mat_scp(self.data_path)


def __main():
    start = time.time()

    train_path, dev_path, test_path = get_dataset_paths(current_dataset, fmt='scp')
    data_train = KaldiDataset(train_path, noise_multiplier=1.0, noise_prob=0.5,
                              supplement_rare_with_noisy=False,
                              supplement_seed=112)
    data_dev = KaldiDataset(dev_path, parent_dataset_path=train_path, training=False)
    data_test = KaldiDataset(test_path, parent_dataset_path=train_path, training=False)

    _print_patients(data_train, data_dev, data_test)

    test = next(data_train.siamese_triplet_epoch(32, augment_parts=True))
    test = next(data_train.siamese_margin_loss_epoch(50, 5))

    print('scp: {0}'.format(time.time() - start))


def __main_snodgrass_test():
    snodgrass_path = '/home/aleks/data/speech_processed/snodgrass_words_cleaned_v3/snodgrass_data_v3.scp'
    data_snodgrass = KaldiDataset(snodgrass_path)

    patients = np.unique([snodgrass_key2patient(x) for x in data_snodgrass.idx2key])
    sessions = np.unique([snodgrass_key2date(x) for x in data_snodgrass.idx2key])
    print(patients)
    print(sessions)
    print(data_snodgrass.data.shape)

    total_seconds_of_data = np.sum(frames2time(x.shape[0]) for x in data_snodgrass.data)
    print('Hours of data: {0:.3f}'.format(total_seconds_of_data / 60 / 60))


def __main_external_test():
    external_path = os.path.join(processed_data_dir, 'external_snodgrass_words.scp')
    data_external = KaldiDataset(external_path)

    total_seconds_of_data = np.sum(frames2time(x.shape[0]) for x in data_external.data)
    print('Hours of data: {0:.3f}'.format(total_seconds_of_data / 60 / 60))


def __main_independent_test():
    swc_path = '/home/aleks/data/speech_processed/independent_test_v2/SWC_independent_test.scp'
    data_swc = KaldiDataset(swc_path)

    print(data_swc.counts)

    train_path, dev_path, test_path = get_dataset_paths('independent_cleaned_v3', fmt='scp')
    data_train = KaldiDataset(train_path)
    data_dev = KaldiDataset(dev_path, parent_dataset_path=train_path)
    data_test = KaldiDataset(test_path, parent_dataset_path=train_path)

    print(data_dev.counts)

    swc_keys = set(data_swc.idx2key)
    dev_keys = set(data_dev.idx2key)
    difference = swc_keys.difference(dev_keys)

    left_words = np.array([key2word(x) for x in difference])
    left_counts = Counter(left_words)
    print(left_counts)


def __dump_numpy_txt():
    import hashlib
    def dump_to_dir(dataset, out_dir, dataset_name):
        if not os.path.exists(os.path.join(out_dir, dataset_name)):
            os.makedirs(os.path.join(out_dir, dataset_name))

        for i, item in enumerate(dataset.data):
            name = hashlib.sha256('{name}{idx}'.format(name=dataset_name, idx=i).encode()).hexdigest() + ".txt"
            word_dir = os.path.join(out_dir, dataset_name, dataset.idx2word[i])
            if not os.path.exists(word_dir):
                os.makedirs(word_dir)
            np.savetxt(os.path.join(word_dir, name), item)

    start = time.time()

    train_path, dev_path, test_path = get_dataset_paths(current_dataset, fmt='scp')
    data_train = KaldiDataset(train_path, noise_multiplier=1.0, noise_prob=0.5,
                              supplement_rare_with_noisy=False,
                              supplement_seed=112)
    dump_to_dir(data_train, current_dataset, 'train')

    data_dev = KaldiDataset(dev_path, parent_dataset_path=train_path, training=False)
    dump_to_dir(data_dev, current_dataset, 'dev')

    data_test = KaldiDataset(test_path, parent_dataset_path=train_path, training=False)
    dump_to_dir(data_test, current_dataset, 'test')

    print('dump: {0}'.format(time.time() - start))


def __dump_lmdb():
    from base.data_io.dataset2lmdb import dataset2lmdb
    start = time.time()

    train_path, dev_path, test_path = get_dataset_paths(current_dataset, fmt='scp')
    train_path_lmdb, dev_path_lmdb, test_path_lmdb = get_dataset_paths(current_dataset, fmt='lmdb')

    data_train = KaldiDataset(train_path, noise_multiplier=1.0, noise_prob=0.5,
                              supplement_rare_with_noisy=False,
                              supplement_seed=112)
    dataset2lmdb(data_train, train_path_lmdb)

    data_dev = KaldiDataset(dev_path, parent_dataset_path=train_path, training=False)
    dataset2lmdb(data_dev, dev_path_lmdb)

    data_test = KaldiDataset(test_path, parent_dataset_path=train_path, training=False)
    dataset2lmdb(data_test, test_path_lmdb)

    print('dump to LMDB: {0}'.format(time.time() - start))


if __name__ == '__main__':
    # __main()
    # __main_snodgrass_test()
    # __main_external_test()
    # __main_independent_test()
    # __dump_numpy_txt()
    __dump_lmdb()
