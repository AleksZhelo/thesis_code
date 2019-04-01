import argparse
import os

from acoustic_word_embeddings.analysis.log_reader import SiameseLogReader, ClassifierLogReader
from base.settings import Settings


def find_run_dirs(folder, verbose=True):
    run_dirs = []
    for subdir in [os.path.join(folder, path) for path in os.listdir(folder) if
                   os.path.isdir(os.path.join(folder, path))]:
        run_name = os.path.basename(subdir)
        conf_path = os.path.join(subdir, 'conf.ini')
        log_path = os.path.join(subdir, 'log')

        if not os.path.exists(conf_path) or not os.path.exists(log_path):
            if verbose:
                print('Not a run dir: {0}, skipping'.format(subdir))
        elif 'siamese' not in run_name and 'classifier' not in run_name:
            if verbose:
                print('Only classifier and siamese runs are supported, skipping {0}'.format(subdir))
        else:
            run_dirs.append(subdir)
    return sorted(run_dirs,
                  key=lambda x: int(os.path.basename(x).split('_')[1]) + (
                      1000000 if 'classifier' in os.path.basename(x) else 0))


def investigate_single(folder):
    print('Looking through {0}'.format(folder))
    run_dirs = find_run_dirs(folder, verbose=True)
    for run_dir in run_dirs:
        run_name = os.path.basename(run_dir)
        conf_path = os.path.join(run_dir, 'conf.ini')
        log_path = os.path.join(run_dir, 'log')
        is_siamese = 'siamese' in run_name

        settings = Settings(conf_path)
        log = SiameseLogReader(log_path) if is_siamese else ClassifierLogReader(log_path)

        if log.epochs.shape[0] == 0:
            print('{0: <80}: no epochs, skipping'.format(run_name))
            continue
        elif log.epochs.shape[0] < 2:
            print('{0: <80}: not enough epochs, skipping'.format(run_name))
            continue
        if not hasattr(settings, 'general_training'):
            print('{0: <80}: format too old, skipping'.format(run_name))
            continue

        use_gru = getattr(settings.general_training, 'use_gru', False)
        noise_multiplier = getattr(settings.general_training, 'noise_multiplier', 0)
        noise_prob = getattr(settings.general_training, 'noise_prob', 0)

        net_base_name = 'GRU_FC_base' if use_gru else 'LSTM_FC_base'
        net_siamese_name = 'SiameseGRU' if use_gru else 'SiameseLSTM'
        net_classifier_name = 'GRUClassifier' if use_gru else 'LSTMClassifier'
        hidden_size_name = 'gru_hidden_size' if use_gru else 'lstm_hidden_size'
        layers_name = 'gru_layers' if use_gru else 'lstm_layers'

        net_base = getattr(settings, net_base_name)
        net_siamese = getattr(settings, net_siamese_name)
        net_classifier = getattr(settings, net_classifier_name)
        custom = getattr(net_base, 'use_custom', False)

        # siamese-specific
        fc_dropout = getattr(net_siamese, 'fc_dropout', [])

        dropout = net_siamese.dropout if is_siamese else net_classifier.dropout
        output_size = net_siamese.output_size if is_siamese else net_classifier.output_size

        net_type_str = 'GRU' if use_gru else 'LSTM'
        out_rnn_part = '{t}({c}) {0}x{1} D{2} {3}'.format(getattr(net_base, layers_name),
                                                          getattr(net_base, hidden_size_name),
                                                          dropout,
                                                          'bi' if net_base.bidirectional else 'uni',
                                                          t=net_type_str,
                                                          c='Custom' if custom else 'Standard')
        out_fc_part = '{0} (D{d})-> {1}'.format(net_base.fc_size, output_size, d=fc_dropout if is_siamese else 'none')
        perf_part = 'epoch {0: <2}: {1:.4f}, epoch {2: <2}: {3:.4f}'.format(
            log.epochs_sorted[-1],
            log.aps_sorted[-1] if is_siamese else log.dev_acc_sorted[-1],
            log.epochs_sorted[-2],
            log.aps_sorted[-2] if is_siamese else log.dev_acc_sorted[-2]  # TODO: skip if only one epoch
        )
        desc = '{0: <30} -> FC {1} noise: {n:.1f}:{p:.1f}'.format(out_rnn_part, out_fc_part, n=noise_multiplier,
                                                                  p=noise_prob)

        print('{name: <80}: {desc: <90} {perf}'.format(desc=desc, name=run_name, perf=perf_part))


def investigate(dirs):
    for d in dirs:
        investigate_single(d)
        print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Iterate over one or more directories with run logs, '
                    'print stats and top 2 performing epochs for each run.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('dir', nargs='+', help='One or more directories with run logs.')
    args = parser.parse_args()

    investigate(args.dir)
