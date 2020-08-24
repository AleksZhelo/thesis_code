import os
import pickle
import random
from collections import Counter
from functools import partial

from rpy2 import robjects
from rpy2.robjects import pandas2ri

from base.common import load_snodgrass_words, get_dataset_paths
from base.data_io.kaldi_dataset import KaldiDataset
from base.util import load_pickled, collapse_nested_dict
from conf import raw_data_dir
from dataset_prep.SWC import aligned_words_file, collect_aligned_words
from dataset_prep.core.common import exclude_words, filter_words_dict


def emu2word_counts(emu_name, emu_dir, seq_rds_path, word_filter, verbose=False):
    if not os.path.exists(seq_rds_path):
        print('Skipping {0} - no segment list found'.format(emu_name))
        return None

    seq_rds = pandas2ri.ri2py(robjects.r.readRDS(seq_rds_path))
    seq_rds = seq_rds.sort_values(by=['labels'])

    selected_words = word_filter(list(seq_rds['labels']))
    seq_filtered = seq_rds.loc[seq_rds['labels'].isin(selected_words)]

    counts = Counter(seq_filtered['labels'].data.obj)
    return counts


def get_emu_word_counts(exclude_snodgrass=True, cache_path='counts_per_db.pckl'):
    def emu2word_counts_except(db, words):
        db_path = os.path.join(raw_data_dir, db)
        seq_rds_path = os.path.join(raw_data_dir, '{0}.rds'.format(db))
        return emu2word_counts(db, db_path, seq_rds_path, partial(exclude_words, words_to_exclude=words),
                               verbose=True)

    snodgrass_words = load_snodgrass_words() if exclude_snodgrass else None
    out_path = cache_path

    if not os.path.exists(out_path):
        emus = filter(lambda x: os.path.isdir(os.path.join(raw_data_dir, x)) and x.endswith('emuDB'),
                      os.listdir(raw_data_dir))
        skip_dbs = ['BROTHERS_emuDB']

        counts_per_db = {}
        for emu in emus:
            if emu not in skip_dbs:
                counts = emu2word_counts_except(emu, snodgrass_words)
                counts_per_db[emu] = counts
        with open(out_path, 'wb') as f:
            pickle.dump(counts_per_db, f)
        return counts_per_db
    else:
        return load_pickled(out_path)


def get_swc_word_counts(exclude_snodgrass=True):
    if not os.path.exists(aligned_words_file):
        collect_aligned_words(verbose=False)

    with open(aligned_words_file, 'rb') as f:
        words_dict = pickle.load(f)

    snodgrass_words = load_snodgrass_words() if exclude_snodgrass else None
    return filter_words_dict(words_dict, word_filter=partial(exclude_words, words_to_exclude=snodgrass_words))


def get_dataset_word_counts(scp_path):
    out_path = os.path.splitext(os.path.basename(scp_path))[0] + '_word_counts.pckl'

    if not os.path.exists(out_path):
        dataset = KaldiDataset(scp_path)
        with open(out_path, 'wb') as f:
            pickle.dump(dataset.counts, f)
        return dataset.counts
    else:
        return load_pickled(out_path)


def select_independent_words():
    """For the new dataset the words are selected as follows: the emuDB datasets are the source of train data, the SWC
    dataset is used as the test set (SWC is read speech only, which should somewhat resemble patient speech: slower and
    more deliberate than typical spontaneous speech).

    The words are selected randomly from words vaguely matching the train dataset words: same or plus one number of
    characters, also nouns (detected by capitalization), and with no fewer test set examples than in the original
    dev set."""
    counts_per_emu_db = get_emu_word_counts()
    counts_emu_total = collapse_nested_dict(counts_per_emu_db)
    counts_swc = get_swc_word_counts()
    train_path, dev_path, _ = get_dataset_paths('all_snodgrass_cleaned_v5', fmt='scp')
    counts_train = get_dataset_word_counts(train_path)
    counts_dev = get_dataset_word_counts(dev_path)

    selected_words = {}
    for word, count in counts_train.items():
        num_letters = len(word)

        emu_similar = {}
        for emu_word, emu_count in counts_emu_total.items():
            if not emu_word.isalpha():  # skipping words with non-alphabetic characters
                continue
            if not emu_word[0].isupper():  # budget noun detector
                continue
            if emu_word in selected_words.values():
                continue
            # factor of 1.5 for the original counts to account for the cleaning removing a lot of examples
            if (len(emu_word) == num_letters or len(emu_word) == num_letters + 1) and emu_count >= count * 1.5:
                emu_similar[emu_word] = emu_count

        similar_list = [x for x in emu_similar.keys() if
                        not any([y.startswith(x) or x.startswith(y) for y in selected_words.values()])]
        random.shuffle(similar_list)
        selected = similar_list[0]
        if word in counts_dev:
            # factor of 1.5 for the original counts to account for the cleaning removing a lot of examples
            while selected not in counts_swc or len(counts_swc[selected]) < counts_dev[word] * 1.5:
                similar_list = similar_list[1:]
                selected = similar_list[0]
                if len(similar_list) == 0:
                    raise RuntimeError()
        selected_words[word] = selected

    return selected_words


def __main():
    selected_words_file = 'selected_words.pckl'
    if not os.path.exists(selected_words_file):
        selected_words = select_independent_words()
        with open(selected_words_file, 'wb') as f:
            pickle.dump(selected_words, f)
    else:
        selected_words = load_pickled(selected_words_file)

    # collect_emu_features('independent_test_v2', 'independent_test', selected_words.values(), debug=False)
    # collect_swc_features('independent_test_v2', 'independent_test', selected_words.values())


if __name__ == '__main__':
    __main()
