import datetime
import glob
import os
import pickle
import re
from shutil import copyfile

import numpy as np

from acoustic_word_embeddings.core.util.args_util import parse_training_args
from acoustic_word_embeddings.core.loss.embedding_loss import DistanceBasedLoss
from acoustic_word_embeddings.core.gru_classifier import GRUClassifier
from acoustic_word_embeddings.core.lstm_classifier import LSTMClassifier
from acoustic_word_embeddings.core.siamese_gru import SiameseGRU
from acoustic_word_embeddings.core.siamese_lstm import SiameseLSTM
from base import util
from base.settings import Settings
from acoustic_word_embeddings.nets.common import torch_load_unwrapped
from base.util import create_logger
from conf import awe_runs_dir


def setup_training_run(model_name):
    args = parse_training_args()
    config = Settings(args.config)

    util.ensure_exists(awe_runs_dir)
    run_name = '{0}_{1}_{2}'.format(
        model_name,
        len([path for path in os.listdir(awe_runs_dir) if os.path.isdir(os.path.join(awe_runs_dir, path))
             and model_name in path]),
        datetime.datetime.now().strftime('%d_%m_%Y')
    )
    log_dir = os.path.join(awe_runs_dir, run_name)
    checkpoint_dir = os.path.join(log_dir, 'checkpoints')
    util.ensure_exists(log_dir)
    util.ensure_exists(checkpoint_dir)

    copyfile(args.config, os.path.join(log_dir, 'conf.ini'))
    logger = create_logger(model_name, os.path.join(log_dir, 'log'))

    logger.info('Running with args:')
    for var in vars(args):
        logger.info('{0}: {1}'.format(var, getattr(args, var)))

    use_gru = config.general_training.use_gru if hasattr(config, 'general_training') else False
    noise_mult = config.general_training.noise_multiplier if hasattr(config.general_training, 'noise_multiplier') else 0
    noise_prob = config.general_training.noise_prob if hasattr(config.general_training, 'noise_prob') else 0
    mean_sub = getattr(config.general_training, 'mean_subtraction', True)
    var_norm = getattr(config.general_training, 'variance_normalization', False)

    return args, config, logger, checkpoint_dir, log_dir, use_gru, noise_mult, noise_prob, mean_sub, var_norm


def load_net(run_dir, epoch=None, logger=None, train=False):
    config = Settings(os.path.join(run_dir, 'conf.ini'))
    use_gru = config.general_training.use_gru if hasattr(config, 'general_training') else False
    mean_sub = getattr(config.general_training, 'mean_subtraction', True)
    var_norm = getattr(config.general_training, 'variance_normalization', False)

    run_name2net = {
        'classifier': LSTMClassifier if not use_gru else GRUClassifier,
        'siamese': SiameseLSTM if not use_gru else SiameseGRU
    }

    run_name: str = os.path.basename(run_dir).split('_')[0]
    run_name = ''.join([x for x in run_name if x.isalpha()])
    net_class = run_name2net[run_name]

    train_scp = read_train_dataset_path(run_dir)
    train_scp_no_ext = os.path.splitext(read_train_dataset_path(run_dir))[0]
    feature_mean = np.load(train_scp_no_ext + '_mean')
    feature_std = np.load(train_scp_no_ext + '_std')
    with open(train_scp_no_ext + '_word2id', 'rb') as f:
        word2id = pickle.load(f)

    loss = None
    if run_name != 'siamese':  # TODO: fragile
        net = net_class(logger, config)
    else:
        loss = create_embedding_loss(config, len(word2id))
        net = net_class(logger, config, loss=loss)
    net.train(train)

    checkpoint_dir = os.path.join(run_dir, 'checkpoints')
    checkpoints = checkpoint_dir2dict(checkpoint_dir)

    if epoch is not None:
        if len(checkpoints) == 0:
            print('No checkpoints found in {0} for run {1}'.format(checkpoint_dir, run_dir))
            return None, None, None, None, None

        if epoch not in checkpoints:
            print('No checkpoint for epoch {0} found in {1} for run {2}'.format(epoch, checkpoint_dir, run_dir))
            return None, None, None, None, None

        net.load_state_dict(torch_load_unwrapped(checkpoints[epoch]))

    return net, config, checkpoints, checkpoint_dir, run_name, loss, train_scp, feature_mean, feature_std, word2id, \
           mean_sub, var_norm


def checkpoint_dir2dict(checkpoint_dir):
    checkpoints = glob.glob(os.path.join(checkpoint_dir, '*.ckpt'))
    checkpoints = {util.checkpoint_path2epoch(x): x for x in checkpoints}
    return checkpoints


def create_embedding_loss(config, num_classes):
    loss_name = config.siamese_training.loss if hasattr(config.siamese_training, 'loss') else 'triplet_loss_offline'
    return DistanceBasedLoss.create(loss_name, config, num_classes)


def read_train_dataset_path(run_dir):
    log_file_path = os.path.join(run_dir, 'log')
    with open(log_file_path, 'r') as f:
        log = f.readlines()

    train_scp = None
    for line in log:
        match = re.search(r'.+Loading (.+), train = True', line)
        if match:
            train_scp = match.group(1)[4:]

    if train_scp is None:
        raise Exception('Failed to find the training dataset path in log file at: {0}'.format(log_file_path))
    return train_scp


def read_embedding_loss(run_dir, throw=True):
    log_file_path = os.path.join(run_dir, 'log')
    with open(log_file_path, 'r') as f:
        log = f.readlines()

    loss_name = None
    for line in log:
        match = re.search(r'.+Using (.+) loss', line)
        if match:
            loss_name = match.group(1)

    if loss_name is None:
        if throw:
            raise Exception(
                'Failed to find embedding loss entry in log file at: {0}, and no default value was given'.format(
                    log_file_path))
    return loss_name
