import numpy as np
from scipy.misc import comb
from scipy.spatial.distance import pdist


def test_chance_level(labels):
    # test_chance_level(np.random.randint(0, 230, 10000))
    # test_chance_level(np.random.randint(0, 500, 10000))
    num_examples = len(labels)
    num_pairs = int(comb(num_examples, 2))

    # build up binary array of matching examples
    matches = np.zeros(num_pairs, dtype=np.bool)

    i = 0
    for n in range(num_examples):
        j = i + num_examples - n - 1
        matches[i:j] = (labels[n] == labels[n + 1:]).astype(np.int32)
        i = j

    num_same = np.sum(matches)

    print(num_same / num_pairs)


def average_precision(data, labels, metric, shuffle_labels=False):
    """
    Adapted from https://github.com/shane-settle/neural-acoustic-word-embeddings/blob/master/code/average_precision.py
    """
    if shuffle_labels:
        labels = labels.copy()
        np.random.seed()
        np.random.shuffle(labels)

    num_examples = len(labels)
    num_pairs = int(comb(num_examples, 2))

    # build up binary array of matching examples
    matches = np.zeros(num_pairs, dtype=np.bool)

    i = 0
    for n in range(num_examples):
        j = i + num_examples - n - 1
        matches[i:j] = (labels[n] == labels[n + 1:]).astype(np.int32)
        i = j

    num_same = np.sum(matches)

    # calculate pairwise distances and sort matches
    dists = pdist(data, metric=metric)
    matches = matches[np.argsort(dists)]

    # calculate precision, average precision, and recall
    precision = np.cumsum(matches) / np.arange(1, num_pairs + 1)
    ap = np.sum(precision * matches) / num_same

    return ap
