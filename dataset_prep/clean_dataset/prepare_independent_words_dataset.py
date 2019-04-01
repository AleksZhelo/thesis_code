import glob
import os
from collections import Counter

import numpy as np

from base.common import get_dataset_paths, key2word
from base.dataset import KaldiDataset
from base.util import load_pickled
from conf import current_dataset, processed_data_dir
from dataset_prep.clean_dataset.collect_independent_words import get_dataset_word_counts


def read_scp_lines(scp_file, line_list):
    with open(scp_file, 'r') as f:
        line_list.extend(f.readlines())


def split_independent_words(output_name, data_sub_dir, dataset_comparable_to):
    output_path = os.path.join(processed_data_dir, output_name)

    train_path, dev_path, _ = get_dataset_paths(dataset_comparable_to)
    counts_train = get_dataset_word_counts(train_path)
    counts_dev = get_dataset_word_counts(dev_path)

    selected_words = load_pickled('selected_words.pckl')

    all_scp = glob.glob(os.path.join(processed_data_dir, data_sub_dir, '*independent_test.scp'))
    swc_scp = [x for x in all_scp if os.path.basename(x).startswith('SWC')][0]
    all_scp.remove(swc_scp)

    emu_lines = []  # this will be the train data
    swc_lines = []  # this will be the test data
    for scp in all_scp:
        read_scp_lines(scp, emu_lines)
    read_scp_lines(swc_scp, swc_lines)
    emu_lines = np.array(emu_lines)
    swc_lines = np.array(swc_lines)

    emu_words = np.array([key2word(x) for x in emu_lines])
    swc_words = np.array([key2word(x) for x in swc_lines])

    emu_counts = Counter(emu_words)
    swc_counts = Counter(swc_words)

    # for word in emu_counts:
    #     print('{0:<20}: train {1}, test {2}'.format(word, emu_counts[word], swc_counts.get(word, 0)))

    # for word in counts_train:
    #     new_word = selected_words[word]
    #     print('{0}, train: {1}, dev: {2}'.format(word, counts_train[word], counts_dev.get(word, 0)))
    #     print('{word}: new train count: {0}, new test count: {1}'.format(emu_counts[new_word], swc_counts[new_word],
    #                                                                      word=new_word))

    new_train = []
    new_dev = []
    for word, new_word in selected_words.items():
        train_new_lines = emu_lines[emu_words == new_word]
        np.random.shuffle(train_new_lines)
        new_train.extend(train_new_lines[:counts_train[word]])

        dev_new_lines = swc_lines[swc_words == new_word]
        np.random.shuffle(dev_new_lines)
        # new_dev.extend(dev_new_lines[:counts_dev.get(word, 5)])  # didn't work at all, maybe bad labels?
        new_dev.extend(dev_new_lines[:35])

    train_scp = '{0}_train.scp'.format(output_path)
    dev_scp = '{0}_dev.scp'.format(output_path)

    with open(train_scp, 'w') as train_file:
        for line in new_train:
            train_file.write(line)

    with open(dev_scp, 'w') as dev_file:
        for line in new_dev:
            dev_file.write(line)

    return train_scp, dev_scp, None


def compose_test_from_non_validation_words(swc_path, dev_path, test_path):
    # and afterwards I switched the dev and test sets, to make sure the test set is the more complete one
    swc_lines = []
    read_scp_lines(swc_path, swc_lines)
    dev_lines = []
    read_scp_lines(dev_path, dev_lines)

    left_lines = np.array([x for x in swc_lines if x not in dev_lines])
    left_words = np.array([key2word(x) for x in left_lines])
    test_lines = []
    for word in np.unique(left_words):
        left_word_lines = left_lines[left_words == word]
        np.random.shuffle(left_word_lines)
        test_lines.extend(left_word_lines[:35])

    with open(test_path, 'w') as test_file:
        for line in test_lines:
            test_file.write(line)


if __name__ == '__main__':
    # train_scp, dev_scp, test_scp = split_independent_words('independent_cleaned_v3', 'independent_test_v2',
    #                                                        'all_snodgrass_cleaned_v4')
    # train_data = KaldiDataset('scp:' + train_scp)
    # train_data.dump_derived_data()

    train_path, dev_path, test_path = get_dataset_paths('independent_cleaned_v3')
    swc_path = '/home/aleks/data/speech_processed/independent_test_v2/SWC_independent_test.scp'

    compose_test_from_non_validation_words(swc_path, dev_path, test_path)
