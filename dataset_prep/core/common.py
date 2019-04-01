import os
from collections import Counter
from functools import partial
from subprocess import call

import numpy as np

from base.common import key2dataset, snodgrass_key2patient
from conf import processed_data_dir


def basic_word_filter(words, counts=None):
    """Filter out non-alpha, short or rare words. Returns a set."""
    word_set = set(words)
    if counts is None:
        counts = Counter(words)
    alpha = [x for x in word_set if x.isalpha()]
    long = [x for x in alpha if len(x) >= 3]
    non_rare = [x for x in long if counts[x] >= 3]
    return non_rare


def select_words(words, words_to_keep):
    """Select only the given words from the word list. Returns a set."""
    word_set = set(words)
    return word_set.intersection(words_to_keep)


def exclude_words(words, words_to_exclude):
    """Select words from the word list absent from the given list. Returns a set."""
    word_set = set(words)
    if words_to_exclude is not None:
        return word_set.difference(words_to_exclude)
    else:
        return word_set


def filter_words_dict(words_dict, word_filter=None):
    words = list(words_dict.keys())
    if word_filter is None:
        word_filter = partial(basic_word_filter, counts={key: len(words_dict[key]) for key in words})
    words_filtered = word_filter(words)
    return {key: words_dict[key] for key in words_filtered}


def fix_scp_encoding(output_name):
    call('iconv -f ISO-8859-15 {0}.scp -t UTF-8 -o {0}.scp_tmp'.format(output_name), shell=True)
    call('mv {0}.scp_tmp {0}.scp'.format(output_name), shell=True)


def scp2snodgrass_patients(scp_path):
    scp_data = []
    with open(scp_path, 'r') as f:
        scp_data.extend(f.readlines())

    patients = [snodgrass_key2patient(line) for line in scp_data if key2dataset(line) == 'snodgrass']
    return [(patient, None) for patient in np.unique(patients)]


def scp2word_lengths_file(scp_path):
    return os.path.join(processed_data_dir, os.path.splitext(os.path.basename(scp_path))[0] + '_word_lengths')
