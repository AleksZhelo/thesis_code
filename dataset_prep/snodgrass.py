import csv
import glob
import os
from collections import namedtuple
from functools import partial
from pprint import pprint

import kaldi_io
import numpy as np
import soundfile

from base import util
from conf import snodgrass_data_dir, processed_data_dir
from dataset_prep.core.cleaning import reject_by_frame_count_snodgrass
from dataset_prep.core.common import fix_scp_encoding
from base.sound_util import time2sample
from dataset_prep.core.features import audio2lmfe

SnodgrassWordRating = namedtuple('SnodgrassWordRating',
                                 ['word', 'order', 'p_score', 'p_delay', 'duration', 'synonym', 'comment', 'vp', 'date',
                                  'wav_path'])


def snodgrass_wav2number(path):
    return int(os.path.basename(path).split('-')[0][6:])


def get_snodgrass_segment(word, vp, date, wav_dict, order, p_delay, duration,
                          append_next=False, channel=0, verbose=False):
    wav_path = wav_dict[order]
    wav_next = wav_dict[order + 1] if (order + 1) in wav_dict else None

    segment_name = '{0}_{1}_snodgrass_{2}_{3}'.format(word, os.path.basename(wav_path).split('-')[0][6:], vp, date)

    # TODO: replace with sound_utils call eventually
    data, rate = soundfile.read(wav_path, always_2d=True)
    sample_start = time2sample(p_delay, rate)
    sample_end = time2sample(p_delay + duration, rate)

    # Append next is either specified explicitly, or inferred from the delay and/or duration not fitting into
    # the current file
    if append_next or sample_start > data.shape[0] or sample_end > data.shape[0]:
        if wav_next is None:
            if verbose:
                print('Invalid segment: {0} - needs the next file to complete the audio, but it is not available'
                      .format(segment_name))
            return None, None, None

        data_next, rate_next = soundfile.read(wav_next, always_2d=True)
        if rate != rate_next:
            if verbose:
                print('Invalid segment: {0} - needs the next file, but it has a different format - not handled for now'
                      .format(segment_name))
            return None, None, None
        data = np.vstack((data, data_next))

    segment_data = data[sample_start:sample_end, channel]
    return segment_name, segment_data, rate


def get_word_audio_segment(word, order, p_score, p_delay, duration, synonym, comment, vp, date, wav_dict, word_counts,
                           verbose):
    # TODO: currently skipping synonyms, but need to include them eventually
    if p_delay != 'nan' and duration != 'nan' and synonym == '':
        if float(p_delay) < 50 and float(duration) < 50 and order in wav_dict:
            segment_name, segment_data, rate = \
                get_snodgrass_segment(word, vp, date, wav_dict, order, p_delay,
                                      duration, append_next=('abgeschnitten' in comment), verbose=verbose)
            if segment_data is not None:
                if word not in word_counts:
                    word_counts[word] = 0
                word_counts[word] += 1

                return segment_name, segment_data, rate


def get_complete_audio(word, order, p_score, p_delay, duration, synonym, comment, vp, date, wav_dict, word_counts,
                       verbose):
    if float(p_delay) < 50 and float(duration) < 50 and order in wav_dict:
        wav_path = wav_dict[order]
        segment_name = '{0}_{1}_snodgrass_{2}_{3}'.format(word, os.path.basename(wav_path).split('-')[0][6:], vp, date)
        segment_data, rate = soundfile.read(wav_path, always_2d=True)
        segment_data = segment_data[:, 0]

        return segment_name, segment_data, rate


def _get_wav_paths_for_rating(order, p_delay, duration, comment, wav_dict):
    wav_path = wav_dict[order]
    wav_next = wav_dict[order + 1] if (order + 1) in wav_dict else None

    info = soundfile.info(wav_path)
    if 'abgeschnitten' in comment or p_delay > info.duration or p_delay + duration > info.duration:
        if wav_next is None:
            return None
        else:
            return [wav_path, wav_next]
    else:
        return wav_path


def get_word_rating(word, order, p_score, p_delay, duration, synonym, comment, vp, date,
                    wav_files_dict, word_counts, verbose):
    if order in wav_files_dict:
        if word not in word_counts:
            word_counts[word] = 0
        word_counts[word] += 1
        return SnodgrassWordRating(word, order, p_score, p_delay, duration, synonym, comment, vp, date,
                                   _get_wav_paths_for_rating(order, p_delay, duration, comment, wav_files_dict))
    else:
        return SnodgrassWordRating(word, 'missing', None, None, None, None, None, None, None, None)


def process_data_dir(vp, date, csv_file, data_dir, word_counts, process_recording, verbose=False):
    word_idx = 0
    p_score_idx = 2
    p_delay_idx = 5
    duration_idx = 6
    synonym_idx = 7
    comment_idx = 8

    wav_files = glob.glob(os.path.join(data_dir, '*.wav'))
    wav_dict = {snodgrass_wav2number(path) - 1: path for path in wav_files}

    output = []
    with open(csv_file, 'r', encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader, None)
        for i, row in enumerate(csv_reader):
            word = row[word_idx]
            p_score = int(row[p_score_idx]) if row[p_score_idx] != '' else None
            p_delay = float(row[p_delay_idx])
            duration = float(row[duration_idx])
            synonym = row[synonym_idx]
            comment = row[comment_idx]

            data = process_recording(word, i, p_score, p_delay, duration, synonym, comment, vp, date, wav_dict,
                                     word_counts, verbose)
            if data is not None:
                output.append(data)

    return output


def snodgrass_iter(snodgrass_dir, process_recording=get_word_audio_segment, verbose=False):
    csv_files = glob.glob(os.path.join(snodgrass_dir, '*.csv'))
    data_dirs = [x for x in os.listdir(snodgrass_dir) if os.path.isdir(os.path.join(snodgrass_dir, x))]
    word_counts = {}

    for file in csv_files:
        parts = os.path.basename(file).split('_')
        vp = parts[0]
        date = parts[1]
        data_dir = [x for x in data_dirs if date in x and x.endswith(vp)][0]

        output = process_data_dir(vp, date, file, os.path.join(snodgrass_dir, data_dir), word_counts, process_recording,
                                  verbose)
        yield data_dir, output, word_counts


def snodgrass2features(input_path, output_path, feature_func, compress=False, verbose=False):
    ark_scp_output = 'ark:| copy-feats --compress={compress} ark:- ark,scp:{0}.ark,{0}.scp' \
        .format(output_path, compress='true' if compress else 'false')

    with kaldi_io.open_or_fd(ark_scp_output, 'wb') as f:
        for data_dir, output, counts in snodgrass_iter(input_path, verbose=verbose):
            for segment_name, segment_data, rate in output:
                features_with_deltas = feature_func(segment_data, rate)

                if reject_by_frame_count_snodgrass(features_with_deltas, segment_name.split('_')[0],
                                                   'snodgrass', segment_name, verbose=True):
                    continue

                kaldi_io.write_mat(f, features_with_deltas, key=segment_name)

    # Fix the encoding of the scp file
    if os.path.exists('{0}.scp'.format(output_path)):
        fix_scp_encoding(output_path)


def print_snodgrass_stats(input_path, verbose=False):
    total_valid = 0
    data_dirs = []
    word_counts = {}

    for data_dir, output, counts in snodgrass_iter(input_path, verbose=verbose):
        data_dirs.append(data_dir)
        total_valid += len(output)
        word_counts = counts

    pprint(word_counts)
    print('Total valid: {0}/{1}'.format(total_valid, len(data_dirs) * 233))


def print_snodgrass_recording_length_stats(input_path, verbose=False):
    lens = []
    for data_dir, output, counts in snodgrass_iter(input_path, process_recording=get_complete_audio, verbose=verbose):
        for key, audio, sample_rate in output:
            lens.append(audio.shape[0] / sample_rate)
    lens = np.array(lens)

    lens = lens[lens > 0.250]  # remove the worst outliers
    print(lens.min(), lens.mean(), lens.max())


if __name__ == '__main__':
    def _process_snodgrass(input_path, sub_dir, feature_func=audio2lmfe):
        sub_dir = os.path.join(processed_data_dir, sub_dir)
        util.ensure_exists(sub_dir)
        output_path = os.path.join(sub_dir, 'snodgrass_data_v3')
        snodgrass2features(input_path, output_path, feature_func, compress=False, verbose=True)


    print_snodgrass_stats(snodgrass_data_dir, verbose=True)
    # print_snodgrass_recording_length_stats(snodgrass_data_dir, verbose=True)
    # _process_snodgrass(snodgrass_data_dir, 'snodgrass_words_cleaned_v3')
    # _process_snodgrass(snodgrass_data_dir, 'snodgrass_words_cleaned_v3_mfcc', feature_func=audio2mfcc)
    # _process_snodgrass(snodgrass_data_dir, 'snodgrass_words_cleaned_v3_reverse', feature_func=audio2lmfe_reverse)
    # _process_snodgrass(snodgrass_data_dir, 'snodgrass_words_cleaned_v3_pncc_mn_off_dct_off',
    #                    feature_func=partial(audio2pncc, mean_norm=False, do_dct=False))

    # print_snodgrass_stats('/home/aleks/data/speech/Naming_Data_Complete/Snodgrass_Recordings_cleaned_denoised',
    #                       verbose=True)
    # _process_snodgrass('/home/aleks/data/speech/Naming_Data_Complete/Snodgrass_Recordings_denoised_19-11-2018',
    #                    'snodgrass_words_cleaned_denoised_19-11-2018')
