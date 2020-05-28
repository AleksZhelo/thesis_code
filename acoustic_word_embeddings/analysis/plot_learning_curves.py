import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc

from acoustic_word_embeddings.analysis.log_reader import ClassifierLogReader, SiameseLogReader


def parse_dir():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--dir',
        type=str,
        required=True,
        help='The dir with runs'
    )

    parser.add_argument(
        '--name',
        type=str,
        required=False,
        help='Run type name to add to save files'
    )

    parser.add_argument(
        '--titles',
        type=str,
        nargs='+',
        required=True,
        help='The titles, one per a triple of runs'
    )

    parser.add_argument(
        '--classifier',
        action='store_true',
        help='Necessary for slightly different log processing',
    )

    parser.add_argument(
        '--dataset',
        type=str,
        required=False,
        help='The dataset to test on'
    )

    return parser.parse_args()


def grouped(container, n):
    return zip(*[iter(container)] * n)


def plot_mean_and_confidence_interval(ax, x, mean, lb, ub, label=None, alpha=0.5,
                                      color_mean=None, color_shading=None):
    # plot the shaded range of the confidence intervals
    # ax.fill_between(x, ub, lb,
    #                 color=color_shading, alpha=.5)
    ax.fill_between(x, ub, lb, alpha=alpha, edgecolor='gray')
    # plot the mean on top
    # mean = savgol_filter(mean, 15, 4)
    ax.plot(x, mean, label=label)


def plot_group(ax, group_metrics, label, alpha):
    n = len(group_metrics)
    mean = np.mean(group_metrics, axis=0)
    std = np.std(group_metrics, axis=0)

    # lower_bound, upper_bound = st.t.interval(0.95, n - 1, loc=mean, scale=std / np.sqrt(n))

    lower_bound, upper_bound = mean - std, mean + std

    plot_mean_and_confidence_interval(
        ax, list(range(group_metrics[0].shape[0])), mean, lower_bound, upper_bound,
        label=label,
        alpha=alpha, color_mean='b', color_shading='b'
    )


def process_run_group(ax, group, title, classifier):
    read_logs = [ClassifierLogReader(os.path.join(x, 'log')) if classifier else SiameseLogReader(os.path.join(x, 'log'))
                 for x in group]

    plot_group(ax, [x.dev_acc if classifier else x.aps for x in read_logs], title, alpha=0.25)


def read_run_triples():
    args = parse_dir()
    runs_root = args.dir
    classifier = args.classifier
    titles = args.titles
    name = args.name
    dataset = args.dataset

    run_dirs = sorted([os.path.join(runs_root, x) for x in os.listdir(runs_root)])
    triples = [[a, b, c] for a, b, c in grouped(run_dirs, 3)]
    return classifier, run_dirs, titles, triples, name, dataset


def __main():
    rc('text', usetex=True)
    rc('font', size=12)
    rc('legend', fontsize=12)
    font = {'family': 'serif', 'serif': ['cmr10']}
    rc('font', **font)

    classifier, run_dirs, titles, triples, name_add, dataset = read_run_triples()

    f, ax = plt.subplots(1, 1, figsize=(9, 5.5))
    if len(triples) == 3:
        ax.set_prop_cycle(
            # color=['xkcd:light grass green', 'xkcd:bright lavender', 'xkcd:cobalt blue', 'xkcd:lightish red'],
            color=['xkcd:bright lavender', 'xkcd:cobalt blue', 'xkcd:lightish red'],
            # dashes=[[5, 5], [5, 1], [1, 1], [1, 0]],  # [1, 5], [3, 5, 1, 5]],
            dashes=[[5, 5], [1, 1], [1, 0]],  # [1, 5], [3, 5, 1, 5]],
            linewidth=[1.5, 1.5, 1.5]
        )
    elif len(triples) == 4:
        ax.set_prop_cycle(
            color=['xkcd:light grass green', 'xkcd:bright lavender', 'xkcd:cobalt blue', 'xkcd:lightish red'],
            dashes=[[5, 5], [5, 1], [1, 1], [1, 0]],
            linewidth=[1.5, 1.5, 1.5, 1.5]
        )
    else:
        ax.set_prop_cycle(
            color=['xkcd:deep sky blue', 'xkcd:light grass green', 'xkcd:bright lavender', 'xkcd:cobalt blue',
                   'xkcd:lightish red'],
            dashes=[[5, 5], [5, 1], [3, 5, 1, 5], [1, 1], [1, 0]],
            linewidth=[1.5, 1.5, 1.5, 1.5, 1.5]
        )
    # ax.set_prop_cycle(
    #     # color=['xkcd:light grass green', 'xkcd:bright lavender', 'xkcd:cobalt blue', 'xkcd:lightish red'],
    #     # color=['xkcd:bright lavender', 'xkcd:cobalt blue', 'xkcd:lightish red'],
    #     # dashes=[[5, 5], [5, 1], [1, 1], [1, 0]],  # [1, 5], [3, 5, 1, 5]],
    #     color=plt.rcParams['axes.prop_cycle'].by_key()['color'][:3],
    #     dashes=[[5, 5], [1, 1], [1, 0]],  # [1, 5], [3, 5, 1, 5]],
    #     linewidth=[1.5, 1.5, 1.5]
    # )

    for i, (triple, title) in enumerate(zip(triples, titles)):
        # if i == 4:
        #     continue
        process_run_group(ax, triple, title, classifier)

    plt.xlabel('Epoch')
    plt.ylabel('Validation accuracy' if classifier else 'Validation AP')

    plt.ylim([0, 80] if classifier else [0, 0.7])

    plt.legend(loc='lower right')

    name = 'classifier_lc' if classifier else 'siamese_lc'
    if name_add is not None:
        name += '_' + name_add
    # plt.show()
    plt.savefig(name + '.pdf', dpi=300, bbox_inches='tight', pad_inches=0)


if __name__ == '__main__':
    __main()
