import argparse
import os

from acoustic_word_embeddings.analysis.log_investigator import find_run_dirs
from acoustic_word_embeddings.analysis.log_reader import SiameseLogReader, ClassifierLogReader
from acoustic_word_embeddings.core.net_util import checkpoint_dir2dict


def remove_checkpoints_single(folder, keep_top=3):
    print('Looking through {0}'.format(folder))
    run_dirs = find_run_dirs(folder, verbose=True)
    for run_dir in run_dirs:
        run_name = os.path.basename(run_dir)
        log_path = os.path.join(run_dir, 'log')
        checkpoint_dir = os.path.join(run_dir, 'checkpoints')
        checkpoints = checkpoint_dir2dict(checkpoint_dir)
        is_siamese = run_name.startswith('siamese')

        log = SiameseLogReader(log_path) if is_siamese else ClassifierLogReader(log_path)

        if log.epochs.shape[0] < 3:
            print('{0: <80}: already fewer than three epochs, skipping'.format(run_name))
        else:
            removed = 0
            for epoch in log.epochs_sorted[:-keep_top]:
                if epoch in checkpoints:
                    # print('Would remove {0}: {1}'.format(epoch, checkpoints[epoch]))
                    removed += 1
                    os.remove(checkpoints[epoch])
            print('Cleaned {0}, removed {1} checkpoints'.format(run_dir, removed))


def remove_checkpoints(dirs, keep_top=3):
    for d in dirs:
        remove_checkpoints_single(d, keep_top=keep_top)
        print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Iterate over one or more directories with run logs, '
                    'remove all but the best 3 checkpoints.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('dir', nargs='+', help='One or more directories with run logs.')
    args = parser.parse_args()

    remove_checkpoints(args.dir, keep_top=3)
