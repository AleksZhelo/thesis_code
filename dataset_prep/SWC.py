import glob
import os
from functools import partial

import kaldi_io
import lxml.etree as etree
import pickle

from collections import OrderedDict

import numpy as np
import soundfile

from base import util
from base.common import load_snodgrass_words
from base.util import print_progress_bar
from conf import raw_data_dir, processed_data_dir, res_dir
from dataset_prep.core.cleaning import reject_by_duration_sec, reject_by_frame_count
from dataset_prep.core.common import select_words, fix_scp_encoding, filter_words_dict
from base.sound_util import time2sample
from dataset_prep.core.features import audio2lmfe, audio2mfcc, audio2lmfe_reverse

corpus_dir = os.path.join(raw_data_dir, 'SWC_German')
aligned_words_file = os.path.join(corpus_dir, 'aligned_words.pkl')


def _parse_aligned_words(swc, words_dict, verbose=False):
    root = etree.parse(swc)
    aligned_words = root.xpath('//t[n][not(n[not(@start)])]')
    if verbose:
        print('Found {0} aligned words'.format(len(aligned_words)))

    if len(root.xpath('//*[@mausinfo]')) > 0:
        pass

    for i, word in enumerate(aligned_words):
        if word.text is not None:
            for normalization in word:
                pronunciation = normalization.attrib['pronunciation']
                start = normalization.attrib['start']
                end = normalization.attrib['end']
                location = swc
                if pronunciation is not None and len(pronunciation) > 0:
                    if pronunciation not in words_dict:
                        words_dict[pronunciation] = []
                    words_dict[pronunciation].append((location, start, end))


def collect_aligned_words(verbose=False):
    words_dict = OrderedDict()
    swc_files = glob.glob(os.path.join(corpus_dir, '*/*.swc'))

    for swc in swc_files:
        if verbose:
            print(swc)
        _parse_aligned_words(swc, words_dict, verbose)

    with open(aligned_words_file, 'wb') as f:
        pickle.dump(words_dict, f, protocol=pickle.HIGHEST_PROTOCOL)


def analyze_aligned_words(words_all_dict, words_filtered_dict):
    def length_ms(keys, words_dict):
        total_length = 0

        for key in keys:
            for item in words_dict[key]:
                total_length += int(item[2]) - int(item[1])

        return total_length

    words_all = np.array(list(words_all_dict.keys()))
    words_filtered = np.array(list(words_filtered_dict.keys()))
    counts = np.array([len(words_filtered_dict[key]) for key in words_filtered_dict])
    print('distinct words: {0}'.format(len(words_filtered_dict)))
    print('mean occurrence count: {0}, min: {1}, median: {2}, max: {3}, total: {4}'.format(
        counts.mean(), counts.min(), np.median(counts), counts.max(), counts.sum()
    ))

    length_filtered = length_ms(words_filtered, words_filtered_dict)
    length_total = length_ms(words_all, words_all_dict)
    print('filtered length (ms): {0}, (h): {1:.2f}'.format(length_filtered, length_filtered / 1000 / 60 / 60))
    print('total length (ms): {0}, (h): {1:.2f}'.format(length_total, length_total / 1000 / 60 / 60))


# TODO: support alphabetic order here?
def swc2features(output_path, swc_dir, feature_func, word_filter=None, channel=0,
                 alphabetic_order=False, compress=False, verbose=False):
    try:
        os.environ['KALDI_ROOT']
    except KeyError:
        print('Please set the KALDI_ROOT variable pointing to a valid Kaldi installation')
        return

    if alphabetic_order:
        print('Alphabetic order currently not supported')
        return

    swc_name = os.path.basename(swc_dir).split('_')[0]

    ark_scp_output = 'ark:| copy-feats --compress={compress} ark:- ark,scp:{0}.ark,{0}.scp' \
        .format(output_path, compress='true' if compress else 'false')
    counter_dict = {}

    with kaldi_io.open_or_fd(ark_scp_output, 'wb') as f:
        article_dirs = list(filter(os.path.isdir, [os.path.join(swc_dir, x) for x in os.listdir(swc_dir)]))
        total = len(article_dirs)
        for n, article_dir in enumerate(article_dirs):
            article_name = os.path.basename(article_dir)
            alignment_file_path = os.path.join(article_dir, 'aligned.swc')
            if not os.path.exists(alignment_file_path):
                print('No alignment file in {0}, skipping'.format(article_dir))
                continue

            words_dict = OrderedDict()
            _parse_aligned_words(alignment_file_path, words_dict, verbose=False)

            words_dict = filter_words_dict(words_dict, word_filter=word_filter)

            if len(words_dict.keys()) == 0:  # don't load sound files if there are no eligible words in the article
                continue

            audio_files = sorted(glob.glob(os.path.join(article_dir, '*.ogg')))
            audio_lengths = [soundfile.info(file).duration for file in audio_files]
            cum_lengths = np.cumsum(audio_lengths)
            audio_data = [soundfile.read(file, always_2d=True) for file in audio_files]

            for word in words_dict:
                if word not in counter_dict:
                    counter_dict[word] = 0
                for segment in words_dict[word]:
                    start_sec = float(segment[1]) / 1000.0
                    end_sec = float(segment[2]) / 1000.0

                    tmp_key = '{0}_{1}_{2}'.format(word, swc_name, article_name)
                    if reject_by_duration_sec(end_sec - start_sec, swc_name, tmp_key, verbose=True):
                        continue

                    file_idx = np.where((cum_lengths > start_sec) == 1)[0][0]
                    if file_idx > 0:  # get the time relative to the correct file
                        start_sec -= cum_lengths[file_idx - 1]
                        end_sec -= cum_lengths[file_idx - 1]
                    rate = audio_data[file_idx][1]

                    if rate < 16000:
                        print('Too low sampling rate for {0}, skipping'.format(audio_files[file_idx]))
                        continue

                    # if rate != 44100:
                    #     print('Wrong sampling rate for {0}, resampling'.format(audio_files[file_idx]))
                    #     resampled = resampy.resample(audio_data[file_idx][0], rate, 44100, filter='kaiser_best', axis=0)
                    #     rate = 44100
                    #     audio_data[file_idx] = (resampled, rate)

                    segment_data = audio_data[file_idx][0][time2sample(start_sec, rate):
                                                           time2sample(end_sec, rate), channel]

                    features_with_deltas = feature_func(segment_data, rate)

                    if reject_by_frame_count(features_with_deltas, word, swc_name, tmp_key, verbose=True):
                        continue

                    kaldi_io.write_mat(f, features_with_deltas,
                                       key='{0}_{1}_{2}_{3}_{4}'.format(word, counter_dict[word],
                                                                        swc_name, article_name,
                                                                        os.path.basename(audio_files[file_idx])))
                    counter_dict[word] += 1
            if verbose:
                print_progress_bar(n, total, prefix='Progress:', suffix='Complete', length=50)
        if verbose:
            print_progress_bar(total, total, prefix='Progress:', suffix='Complete', length=50)

    # Fix the encoding of the scp file
    if os.path.exists('{0}.scp'.format(output_path)):
        fix_scp_encoding(output_path)


def collect_swc_features(processed_sub_dir, collection_name, word_list, feature_func=audio2lmfe):
    sub_dir = os.path.join(processed_data_dir, processed_sub_dir)
    util.ensure_exists(sub_dir)
    output_name = os.path.join(sub_dir, 'SWC_{0}'.format(collection_name))
    swc2features(output_name, corpus_dir, feature_func, word_filter=partial(select_words, words_to_keep=word_list),
                 verbose=True)


def __main():
    if not os.path.exists(aligned_words_file):
        collect_aligned_words(verbose=True)

    with open(aligned_words_file, 'rb') as f:
        words_dict = pickle.load(f)

    snodgrass_words = load_snodgrass_words()

    snodgrass_words_dict = filter_words_dict(words_dict,
                                             word_filter=partial(select_words, words_to_keep=snodgrass_words))
    analyze_aligned_words(words_dict, snodgrass_words_dict)
    analyze_aligned_words(words_dict, filter_words_dict(words_dict))

    collect_swc_features('snodgrass_words_cleaned_v3', 'snodgrass_words', snodgrass_words)
    # collect_swc_features('snodgrass_words_cleaned_v3_mfcc', 'snodgrass_words', snodgrass_words,
    #                      feature_func=audio2mfcc)
    # collect_swc_features('snodgrass_words_cleaned_v3_reverse', 'snodgrass_words', snodgrass_words,
    #                      feature_func=audio2lmfe_reverse)

    # with open(os.path.join(res_dir, 'new_words.txt'), 'r') as f:
    #     new_words = [line.rstrip('\n') for line in f]
    # collect_swc_features('new_words', 'new_words', new_words)


if __name__ == '__main__':
    __main()
