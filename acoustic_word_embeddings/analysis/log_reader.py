import re

import numpy as np


class SiameseLogReader(object):

    def __init__(self, path):
        with open(path, 'r') as f:
            lines = f.readlines()

        epochs = []
        aps = []
        for line in lines:
            match = re.search(r'.+Epoch ([0-9]+) avg dev precision= ([0-9]\.[0-9]+)', line)
            if match:
                epochs.append(int(match.group(1)))
                aps.append(float(match.group(2)))

        self.epochs = np.array(epochs)
        self.aps = np.array(aps)  # this is dev APs

        perf_idx = np.argsort(self.aps)
        self.epochs_sorted = self.epochs[perf_idx]
        self.aps_sorted = self.aps[perf_idx]


class ClassifierLogReader(object):

    def __init__(self, path):
        with open(path, 'r') as f:
            lines = f.readlines()

        epochs = []
        training_acc = []
        dev_acc = []
        for line in lines:
            match = re.search(r'.+Epoch ([0-9]+) training accuracy = ([0-9]+\.[0-9]+), dev accuracy = ([0-9]+\.[0-9]+)',
                              line)
            if match:
                epochs.append(int(match.group(1)))
                training_acc.append(float(match.group(2)))
                dev_acc.append(float(match.group(3)))

        self.epochs = np.array(epochs)
        self.training_acc = np.array(training_acc)
        self.dev_acc = np.array(dev_acc)

        perf_idx = np.argsort(self.dev_acc)
        self.epochs_sorted = self.epochs[perf_idx]
        self.training_acc_sorted = self.training_acc[perf_idx]
        self.dev_acc_sorted = self.dev_acc[perf_idx]


if __name__ == '__main__':
    test_classifier = ClassifierLogReader(
        '/home/aleks/projects/thesis/acoustic_word_embeddings/runs_cluster/classifier_7_03_11_2018/log'
    )
    print(test_classifier.epochs_sorted)
    print(test_classifier.training_acc_sorted)
    print(test_classifier.dev_acc_sorted)
