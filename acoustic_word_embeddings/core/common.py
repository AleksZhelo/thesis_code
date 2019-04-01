import glob
import os
import pickle
from collections import Counter

import numpy as np

from base.common import key2word, key2dataset


def data_path_to_epoch(path):
    return int(os.path.splitext(path)[0].split('_')[-1])


def embeddings_dir2dict(embeddings_dir):
    return {data_path_to_epoch(path): path for path in glob.glob(os.path.join(embeddings_dir, '*.pickle'))}


def load_embeddings(path, data_name='dev', return_keys=False):
    with open(path, 'rb') as f:
        data_dict = pickle.load(f)

    words = np.array([key2word(key) for key in data_dict])
    datasets = np.array([key2dataset(key) for key in data_dict])
    datasets[datasets == 'PHATTSESSIONZ'] = 'PHA'
    vecs = np.array([vec for vec in data_dict.values()])
    counts = Counter(words)
    word_idxs = {key: np.where(words == key)[0] for key in counts}
    print('There are {0} unique words in the {1} set.'.format(len(counts), data_name))

    if not return_keys:
        return words, datasets, vecs, counts, word_idxs
    else:
        return words, datasets, vecs, counts, word_idxs, np.array(list(data_dict.keys()))
