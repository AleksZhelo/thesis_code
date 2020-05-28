import os
from functools import partial

import kaldi_io
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri

from base import util, sound_util
from base.common import load_snodgrass_words
from base.util import print_progress_bar
from conf import raw_data_dir, processed_data_dir
from dataset_prep.core.WavCache import WavCache
from dataset_prep.core.cleaning import reject_by_duration_sec, reject_by_frame_count
from dataset_prep.core.common import basic_word_filter, select_words, fix_scp_encoding
from dataset_prep.core.features import audio2lmfe


# TODO: log all the data cleaning messages
def emu2features(output_path, emu_name, emu_dir, seq_rds_path, feature_func, word_filter=None,
                 alphabetic_order=True, compress=False, verbose=False):
    try:
        os.environ['KALDI_ROOT']
    except KeyError:
        print('Please set the KALDI_ROOT variable pointing to a valid Kaldi installation')
        return

    wav_files = WavCache(maxsize=1000 * 2 ** 20)
    ark_scp_output = 'ark:| copy-feats --compress={compress} ark:- ark,scp:{0}.ark,{0}.scp' \
        .format(output_path, compress='true' if compress else 'false')

    if not os.path.exists(seq_rds_path):
        print('Skipping {0} - no segment list found'.format(emu_name))
        return

    seq_rds = pandas2ri.ri2py(robjects.r.readRDS(seq_rds_path))
    seq_rds = seq_rds.sort_values(by=['labels'])

    if word_filter is None:
        word_filter = basic_word_filter

    selected_words = word_filter(list(seq_rds['labels']))
    seq_filtered = seq_rds.loc[seq_rds['labels'].isin(selected_words)]

    if not alphabetic_order:
        seq_filtered = seq_filtered.sort_values(by=['session', 'bundle'])  # improve cache performance
        # don't even need the cache then, the segments are sorted so that the wav-files are processed sequentially

    total = len(seq_filtered.index)
    if total > 0:
        print('Selected {0} segments from the {1} dataset'.format(total, emu_name))
    else:
        print('No matching segments found in {0}, exiting'.format(emu_name))
        return

    n = 0
    with kaldi_io.open_or_fd(ark_scp_output, 'wb') as f:
        for i, row in seq_filtered.iterrows():
            word = row['labels']
            session = row['session']
            bundle = row['bundle']
            key = '{0}_{1}_{2}_{3}_{4}'.format(word, i, emu_name.split('_')[0], session, bundle)
            duration_sec = (row['end'] - row['start']) / 1000

            if reject_by_duration_sec(duration_sec, emu_name, key, verbose=True):
                continue

            wav_path = os.path.join(emu_dir, '{0}_ses'.format(session), '{0}_bndl'.format(bundle),
                                    '{0}.wav'.format(bundle))

            rate, signal = wav_files[wav_path]
            segment = signal[row['sample_start']:row['sample_end']]
            if rate < 16000:
                print('Too low sampling rate for {0}, skipping'.format(key))
                continue
            # segment = signal[time2sample(row['start'] / 1000, rate):time2sample(row['end'] / 1000, rate)]

            features_with_deltas = feature_func(segment, rate)

            if reject_by_frame_count(features_with_deltas, word, emu_name, key, verbose=True):
                continue

            kaldi_io.write_mat(f, features_with_deltas, key=key)

            n += 1
            if verbose and n % 100 == 0:
                print_progress_bar(n, total, prefix='Progress:', suffix='Complete', length=50)
    if verbose:
        print_progress_bar(total, total, prefix='Progress:', suffix='Complete', length=50)

    # Fix the encoding of the scp file
    if os.path.exists('{0}.scp'.format(output_path)):
        fix_scp_encoding(output_path)


def emu2wav_segments(output_path, emu_name, emu_dir, seq_rds_path, word_filter=None,
                     alphabetic_order=True, compress=False, verbose=False):
    wav_files = WavCache(maxsize=1000 * 2 ** 20)

    if not os.path.exists(seq_rds_path):
        print('Skipping {0} - no segment list found'.format(emu_name))
        return

    seq_rds = pandas2ri.ri2py(robjects.r.readRDS(seq_rds_path))
    seq_rds = seq_rds.sort_values(by=['labels'])

    if word_filter is None:
        word_filter = basic_word_filter

    selected_words = word_filter(list(seq_rds['labels']))
    seq_filtered = seq_rds.loc[seq_rds['labels'].isin(selected_words)]

    if not alphabetic_order:
        seq_filtered = seq_filtered.sort_values(by=['session', 'bundle'])  # improve cache performance
        # don't even need the cache then, the segments are sorted so that the wav-files are processed sequentially

    total = len(seq_filtered.index)
    if total > 0:
        print('Selected {0} segments from the {1} dataset'.format(total, emu_name))
    else:
        print('No matching segments found in {0}, exiting'.format(emu_name))
        return

    util.ensure_exists(output_path)
    n = 0
    for i, row in seq_filtered.iterrows():
        if n > 1000:
            break
        word = row['labels']
        session = row['session']
        bundle = row['bundle']
        key = '{0}_{1}_{2}_{3}_{4}'.format(word, i, emu_name.split('_')[0], session, bundle)
        duration_sec = (row['end'] - row['start']) / 1000

        if reject_by_duration_sec(duration_sec, emu_name, key, verbose=True):
            continue

        wav_path = os.path.join(emu_dir, '{0}_ses'.format(session), '{0}_bndl'.format(bundle),
                                '{0}.wav'.format(bundle))

        rate, signal = wav_files[wav_path]
        segment = signal[row['sample_start']:row['sample_end']]
        if rate < 16000:
            print('Too low sampling rate for {0}, skipping'.format(key))
            continue
        # segment = signal[time2sample(row['start'] / 1000, rate):time2sample(row['end'] / 1000, rate)]

        features_with_deltas = audio2lmfe(segment, rate)

        if reject_by_frame_count(features_with_deltas, word, emu_name, key, verbose=True):
            continue

        sound_util.write_array_to_wav(os.path.join(output_path, key), segment, rate)

        n += 1
        if verbose and n % 100 == 0:
            print_progress_bar(n, total, prefix='Progress:', suffix='Complete', length=50)
    if verbose:
        print_progress_bar(total, total, prefix='Progress:', suffix='Complete', length=50)


def collect_emu_features(processed_sub_dir, collection_name, word_list, feature_func=audio2lmfe, debug=False):
    def _process_emu_db(db, sub_dir, name, words):
        sub_dir = os.path.join(processed_data_dir, sub_dir)
        util.ensure_exists(sub_dir)
        db_path = os.path.join(raw_data_dir, db)
        seq_rds_path = os.path.join(raw_data_dir, '{0}.rds'.format(db))
        if debug:
            emu2wav_segments(os.path.join(sub_dir, '{0}_{1}'.format(db, name)),
                             db, db_path, seq_rds_path, partial(select_words, words_to_keep=words),
                             verbose=True)
        else:
            emu2features(os.path.join(sub_dir, '{0}_{1}'.format(db, name)),
                         db, db_path, seq_rds_path, feature_func, partial(select_words, words_to_keep=words),
                         verbose=True)

    emus = filter(lambda x: os.path.isdir(os.path.join(raw_data_dir, x)) and x.endswith('emuDB'),
                  os.listdir(raw_data_dir))
    skip_dbs = ['BROTHERS_emuDB']
    for emu in emus:
        if emu not in skip_dbs:
            _process_emu_db(emu, processed_sub_dir, collection_name, word_list)
    # _process_emu_db('PD1_emuDB')


if __name__ == '__main__':
    snodgrass_words = load_snodgrass_words()
    collect_emu_features('snodgrass_words_cleaned_v3', 'snodgrass_words', snodgrass_words)
    # collect_emu_features('snodgrass_words_cleaned_v3_mfcc', 'snodgrass_words', snodgrass_words, feature_func=audio2mfcc)
    # collect_emu_features('snodgrass_words_cleaned_v3_reverse', 'snodgrass_words', snodgrass_words,
    #                      feature_func=audio2lmfe_reverse)
    # collect_emu_features('snodgrass_words_cleaned_v3_pncc_mn_off_dct_off', 'snodgrass_words', snodgrass_words,
    #                      feature_func=partial(audio2pncc, mean_norm=False, do_dct=False))

    # collect_emu_features('test_ahm', 'test_ahm', ['<ähm>'], feature_func=None, debug=True)

    # with open(os.path.join(res_dir, 'new_words.txt'), 'r') as f:
    #     new_words = [line.rstrip('\n') for line in f]
    # collect_emu_features('new_words', 'new_words', new_words)
