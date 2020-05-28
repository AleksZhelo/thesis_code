import glob
import os
from collections import Counter

import numpy as np

from base.common import get_dataset_paths, key2word, key2dataset, snodgrass_key2patient
from base.dataset import KaldiDataset
from base.knapsack import knapsack
from base.util import remove_all
from conf import processed_data_dir
from dataset_prep.core.common import scp2snodgrass_patients


def split_snodgrass_dataset(source_sub_dir, snodgrass_file, same_split_as=None):
    """~50% train, ~25% percent dev, ~25% test, while taking care that each patient's data is present exclusively
     in either train, dev, or test sets."""
    snodgrass = os.path.join(processed_data_dir, source_sub_dir, snodgrass_file)
    lines = []
    with open(snodgrass, 'r') as f:
        lines.extend(f.readlines())

    lines = np.array(lines)
    words = np.array([key2word(x) for x in lines])
    patients = np.array([snodgrass_key2patient(x) for x in lines])
    patients_with_counts = [(key, value) for key, value in Counter(patients).items()]

    data_train = []
    data_test = []
    data_dev = []

    words_train = []
    words_test = []
    words_dev = []

    if same_split_as is None:
        # surprise knapsack problem :)
        patients_train = knapsack(patients_with_counts, len(lines) / 2)[1]
        patients_left = remove_all(patients_with_counts, patients_train)
        patients_test = knapsack(patients_left, len(lines) / 4)[1]
        patients_dev = remove_all(patients_left, patients_test)
    else:
        train_path, dev_path, test_path = get_dataset_paths(same_split_as)
        patients_train = scp2snodgrass_patients(train_path)
        patients_test = scp2snodgrass_patients(test_path)
        patients_dev = scp2snodgrass_patients(dev_path)

    for patient, _ in patients_train:
        data_train.extend(lines[np.where(patients == patient)])
        words_train.extend(words[np.where(patients == patient)])

    for patient, _ in patients_test:
        data_test.extend(lines[np.where(patients == patient)])
        words_test.extend(words[np.where(patients == patient)])

    for patient, _ in patients_dev:
        data_dev.extend(lines[np.where(patients == patient)])
        words_dev.extend(words[np.where(patients == patient)])

    print('Unique words in train dataset: {0}, in test: {1}, in dev: {2}'
          .format(len(Counter(words_train)), len(Counter(words_test)), len(Counter(words_dev))))

    return data_train, data_dev, data_test


def split_train_dev_test(output_name, external_sub_dir, snodgrass_sub_dir, snodgrass_file, same_split_as=None):
    output_path = os.path.join(processed_data_dir, output_name)
    external_snodgrass = glob.glob(os.path.join(processed_data_dir, external_sub_dir, '*snodgrass_words.scp'))

    lines = []
    for scp in external_snodgrass:
        with open(scp, 'r') as f:
            lines.extend(f.readlines())

    words = np.array([key2word(line) for line in lines])
    datasets = np.array([key2dataset(line) for line in lines])
    counts = Counter(words)

    word_dataset2idx = {key: {dset: [] for dset in np.unique(datasets)} for key in counts}
    for i in range(len(lines)):
        word_dataset2idx[words[i]][datasets[i]].append(i)

    idx_train = []
    idx_dev = []
    idx_train.extend(range(len(lines)))

    snodgrass_train, snodgrass_dev, snodgrass_test = split_snodgrass_dataset(snodgrass_sub_dir, snodgrass_file,
                                                                             same_split_as=same_split_as)

    train_scp = '{0}_train.scp'.format(output_path)
    dev_scp = '{0}_dev.scp'.format(output_path)
    test_scp = '{0}_test.scp'.format(output_path)

    with open(train_scp, 'w') as train_file:
        for idx in idx_train:
            train_file.write(lines[idx])
        for line in snodgrass_train:
            train_file.write(line)

    with open(dev_scp, 'w') as dev_file:
        for idx in idx_dev:
            dev_file.write(lines[idx])
        for line in snodgrass_dev:
            dev_file.write(line)

    with open(test_scp, 'w') as test_file:
        for line in snodgrass_test:
            test_file.write(line)

    return train_scp, dev_scp, test_scp


def __main_v5():
    train_scp, dev_scp, test_scp = \
        split_train_dev_test(output_name='all_snodgrass_cleaned_v5',
                             external_sub_dir='snodgrass_words_cleaned_v3',
                             snodgrass_sub_dir='snodgrass_words_cleaned_v3',
                             snodgrass_file='snodgrass_data_v3.scp',
                             same_split_as='all_snodgrass_cleaned_v3')

    # Dump the training dataset mean and word2id for future use
    train_data = KaldiDataset('scp:' + train_scp)
    train_data.dump_derived_data()


if __name__ == '__main__':
    __main_v5()
