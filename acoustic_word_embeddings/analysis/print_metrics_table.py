import os

import numpy as np
import pandas as pd

from acoustic_word_embeddings.analysis.log_reader import ClassifierLogReader, SiameseLogReader
from acoustic_word_embeddings.analysis.plot_learning_curves import read_run_triples
from acoustic_word_embeddings.calculate_accuracy import do_calculate_accuracy
from acoustic_word_embeddings.calculate_ap import do_calculate_ap


def process_run_group(group, classifier, dataset):
    read_logs = [ClassifierLogReader(os.path.join(x, 'log')) if classifier else SiameseLogReader(os.path.join(x, 'log'))
                 for x in group]

    results = []
    for run, log in zip(group, read_logs):
        epoch = np.argmax(log.dev_acc if classifier else log.aps)
        results.append(calculate_metrics(run, epoch, classifier, dataset=dataset))

    mean = np.mean(results, axis=0)
    std = np.std(results, axis=0)

    return mean, std


def calculate_metrics(run_dir, epoch, classifier, dataset=None):
    dev_ap = do_calculate_ap(run_dir, epoch, dataset=dataset, partition='dev')
    test_ap = do_calculate_ap(run_dir, epoch, dataset=dataset, partition='test')

    dev_accuracy = do_calculate_accuracy(run_dir, epoch, classifier, dataset=dataset, partition='dev')
    test_accuracy = do_calculate_accuracy(run_dir, epoch, classifier, dataset=dataset, partition='test')

    # print("Dev AP: {0:.3f}, test AP: {1:.3f}".format(dev_ap, test_ap))
    # print("Dev acc: {0:.3f}, test acc: {1:.3f}".format(dev_accuracy, test_accuracy))

    return [dev_ap, test_ap, dev_accuracy, test_accuracy]


def mean_plus_minus_std(means, stds, column):
    return ['${0:.3f}\pm{1:.3f}$'.format(x, y) for x, y in zip(means[:, column], stds[:, column])]


def __main():
    classifier, run_dirs, titles, triples, name, dataset = read_run_triples()

    means = []
    stds = []
    for triple in triples:
        mean, std = process_run_group(triple, classifier, dataset)
        means.append(mean)
        stds.append(std)

    means = np.array(means)
    stds = np.array(stds)

    data = {
        'Val AP': mean_plus_minus_std(means, stds, 0),
        'Test AP': mean_plus_minus_std(means, stds, 1),
        'Val accuracy': mean_plus_minus_std(means, stds, 2),
        'Test accuracy': mean_plus_minus_std(means, stds, 3),
    }
    dataframe = pd.DataFrame(data, index=titles)
    print(dataframe.to_latex(escape=False))


if __name__ == '__main__':
    __main()

    # # run_dir = '/home/aleks/projects/thesis/acoustic_word_embeddings/runs/siamese_22_19_12_2018'
    # # epoch = 76
    # run_dir = '/home/aleks/projects/thesis/acoustic_word_embeddings/runs_cluster/siamese_53_20_12_2018'
    # epoch = 66  # 66, 83
    #
    # # res = calculate_metrics(run_dir, epoch, False, dataset='independent_cleaned_v3')
    # res = calculate_metrics(run_dir, epoch, False, dataset='all_snodgrass_cleaned_v5')
    # print(res)
