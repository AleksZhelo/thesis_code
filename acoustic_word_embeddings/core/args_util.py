import argparse


def _add_run_dir(parser):
    parser.add_argument(
        '--run_dir',
        type=str,
        required=True,
        help='The directory with the finished run data'
    )


def _add_run_epoch(parser):
    parser.add_argument(
        '--run_epoch',
        type=int,
        required=True,
        help='The epoch of the run'
    )


def _add_ratings_file(parser):
    parser.add_argument(
        '--ratings_file',
        type=str,
        required=True,
        help='The ratings index file to use'
    )


def _add_net_annotations_file(parser):
    parser.add_argument(
        '--net_annotations_file',
        type=str,
        required=True,
        help='The net annotations file to use'
    )


def _add_beta_file(parser, help_str=None):
    if help_str is None:
        help_str = 'The file with the beta margin values to use'
    parser.add_argument(
        '--beta_file',
        type=str,
        required=True,
        help=help_str
    )


def _add_word2id_file(parser):
    parser.add_argument(
        '--word2id_file',
        type=str,
        required=True,
        help='The file with the word to numerical id mapping to use'
    )


def _add_word_lengths_file(parser):
    parser.add_argument(
        '--word_lengths_file',
        type=str,
        required=True,
        help='The file with the word duration statistics'
    )


def _add_patient_dir(parser):
    parser.add_argument(
        '--patient_dir',
        type=str,
        required=True,
        help='The directory with one or more sessions of a patient.'
    )


def _add_auto_rating_args(parser):
    _add_run_dir(parser)
    _add_run_epoch(parser)
    _add_ratings_file(parser)


def parse_training_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='The config file with the network and training settings'
    )

    parser.add_argument(
        '--load_weights',
        type=str,
        required=False,
        help='The checkpoint to load pre-trained weights from'
    )

    parser.add_argument(
        '--gpu_count',
        type=int,
        required=False,
        default=1,
        help='The number of GPUs to use'
    )

    return parser.parse_args()


def parse_gen_args():
    parser = argparse.ArgumentParser()

    _add_run_dir(parser)

    return parser.parse_args()


def parse_auto_rating_args():
    parser = argparse.ArgumentParser()

    _add_auto_rating_args(parser)
    parser.add_argument(
        '--vad',
        action='store_true',
        help='Use VAD instead of network-based segmentation',
    )

    return parser.parse_args()


def parse_rate_new_data_args():
    parser = argparse.ArgumentParser()

    _add_patient_dir(parser)
    _add_run_dir(parser)
    _add_run_epoch(parser)
    _add_word2id_file(parser)

    return parser.parse_args()


def parse_load_epoch_args():
    parser = argparse.ArgumentParser()

    _add_run_dir(parser)
    _add_run_epoch(parser)

    return parser.parse_args()


def parse_false_positives_args():
    parser = argparse.ArgumentParser()

    _add_ratings_file(parser)
    _add_net_annotations_file(parser)
    _add_beta_file(parser)
    _add_word2id_file(parser)

    return parser.parse_args()


def parse_rerating_args():
    parser = argparse.ArgumentParser()

    _add_auto_rating_args(parser)
    _add_net_annotations_file(parser)
    _add_beta_file(parser, help_str='The file with the original network\'s beta margin values')
    _add_word2id_file(parser)
    _add_word_lengths_file(parser)

    return parser.parse_args()


def parse_patient_adaptation_test_args():
    parser = argparse.ArgumentParser()

    _add_auto_rating_args(parser)

    parser.add_argument(
        '--patient',
        type=str,
        required=True,
        help='The patient to test on'
    )

    return parser.parse_args()
