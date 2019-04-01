import numpy as np
from scipy.spatial.distance import cdist

from acoustic_word_embeddings.core.common import load_embeddings
from acoustic_word_embeddings.gen_embeddings import get_or_generate_embeddings


def top_3_candidates(x_embedding, vecs_train, word_idxs_train, verbose=False):
    dists_dev = []
    words_tested = []
    for word in word_idxs_train:
        dists_x = cdist(x_embedding[np.newaxis, :], vecs_train[word_idxs_train[word]], metric='cosine')
        dists_dev.append(np.mean(dists_x))
        words_tested.append(word)
    words_tested = np.array(words_tested)
    dists_dev = np.array(dists_dev)
    min_idx = np.argsort(dists_dev)[:3]
    if verbose:
        print(", ".join(['{0}: {1:.2f}'.format(word, dist) for word, dist in zip(words_tested[min_idx],
                                                                                 dists_dev[min_idx])]))
    return words_tested[min_idx]


def classify_by_avg_distance(run_dir, epoch):
    train_embeddings, dev_embeddings, _ = get_or_generate_embeddings(run_dir, epoch, dev_needed=True)
    words_train, datasets_train, vecs_train, counts_train, word_idxs_train = load_embeddings(
        train_embeddings[epoch])
    words_dev, datasets_dev, vecs_dev, counts_dev, word_idxs_dev = load_embeddings(dev_embeddings[epoch])

    correct = 0
    total = 0
    for word in word_idxs_dev:
        for i in range(len(word_idxs_dev[word])):
            dev_x = vecs_dev[word_idxs_dev[word][i]]
            res = top_3_candidates(dev_x, vecs_train, word_idxs_train)
            if res[0] == word:
                correct += 1
            total += 1

    print('Correct {0}/{1} ({2:.2f}% accuracy)'.format(correct, total, (correct / total) * 100))


def __main():
    run_dir = '../acoustic_word_embeddings/runs_cluster/siamese_17_08_11_2018'
    epoch = 92
    classify_by_avg_distance(run_dir, epoch)


if __name__ == '__main__':
    __main()
