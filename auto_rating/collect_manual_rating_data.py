import os
import pickle
import sys
from shutil import copy
from typing import List

import numpy as np
import resampy
import soundfile

from base import util
from base.common import snodgrass_key2patient, snodgrass_key2all, response_missing
from base.kaldi_dataset import KaldiDataset
from conf import snodgrass_data_dir, processed_data_dir
from dataset_prep.snodgrass import snodgrass_iter, get_word_rating, SnodgrassWordRating


def get_snodgrass_ratings_for_keys(dataset: KaldiDataset):
    snodgrass_ratings = {}
    for data_dir, output, _ in snodgrass_iter(snodgrass_data_dir, process_recording=get_word_rating):
        key = '{vp}_{date}'.format(vp=data_dir.split('_')[-1], date=data_dir.split('_')[0])
        snodgrass_ratings[key] = output

    ratings = []
    for i in range(dataset.data.shape[0]):
        key = dataset.idx2key[i]
        _, order, _, vp, date = snodgrass_key2all(key)

        rating_key = '{vp}_{date}'.format(vp=vp, date=date)
        if rating_key in snodgrass_ratings:
            rating = snodgrass_ratings[rating_key][order]
            ratings.append(rating)
    return ratings


def get_snodgrass_ratings_for_patients(dataset: KaldiDataset):
    # I don't split the data of the same patient between datasets, so the date doesn't matter here
    patients = set([snodgrass_key2patient(key) for key in dataset.idx2key])

    ratings = []
    for data_dir, output, _ in snodgrass_iter(snodgrass_data_dir, process_recording=get_word_rating):
        key = '{vp}'.format(vp=data_dir.split('_')[-1])
        if key in patients:
            ratings.extend(output)

    return ratings


def collect_csv_ratings(dataset_scp, include_missing=True):
    output_name = os.path.splitext(dataset_scp)[0] + '_ratings' + '_full' if include_missing else ''
    data_path = os.path.join(processed_data_dir, dataset_scp)
    # training set to True to avoid providing parent path
    dataset = KaldiDataset('scp:' + data_path, training=True, logger=None)

    ratings = get_snodgrass_ratings_for_keys(dataset) if not include_missing \
        else get_snodgrass_ratings_for_patients(dataset)
    ratings_cleaned = [rating for rating in ratings if rating.wav_path is not None]
    with open(os.path.join(processed_data_dir, output_name), 'wb') as f:
        pickle.dump(ratings_cleaned, f, protocol=pickle.HIGHEST_PROTOCOL)

    return output_name


def update_ratings_with_fixed_audio(ratings_file):
    """Collects the sound data for words where the response spills to the next recording, also performs resampling
    to 32000 for VAD (which can only work with the sampling rate in multiples of 8000).
     After resampling the audio for VAD can be additionally cleaned with sox:
    find . -name '*_32.wav' | parallel 'sox {} {.}-clean.wav noisered ~/noise.profile 0.2'"""

    fixed_audio_output_dir = os.path.join(processed_data_dir, os.path.basename(ratings_file) + '_audio')
    util.ensure_exists(fixed_audio_output_dir)
    with open(os.path.join(processed_data_dir, ratings_file), 'rb') as f:
        ratings: List[SnodgrassWordRating] = pickle.load(f)

    ratings_updated = []
    for rating in ratings:
        new_name = '{0}_{1}_{2}_{3}.wav'.format(rating.word, rating.order, rating.vp, rating.date)
        new_name_32 = '{0}_{1}_{2}_{3}_32.wav'.format(rating.word, rating.order, rating.vp, rating.date)
        target_path = os.path.join(fixed_audio_output_dir, new_name)
        target_path_32 = os.path.join(fixed_audio_output_dir, new_name_32)

        if not isinstance(rating.wav_path, list):
            copy(rating.wav_path, target_path)

            audio, rate = soundfile.read(rating.wav_path)
            data_32 = resampy.resample(audio[:, 0], rate, 32000, filter='kaiser_best')
            soundfile.write(target_path_32, data_32, 32000)
        else:
            data = []
            rates = []
            for wav_path in rating.wav_path:
                audio, rate = soundfile.read(wav_path)
                data.append(audio)
                rates.append(rate)

            if not all([x == rates[0] for x in rates]):
                print('Unequal sampling rates for snodgrass files not supported')
                sys.exit(-1)
            else:
                data = np.concatenate(data, axis=0)
                soundfile.write(target_path, data, rates[0])

                data_32 = resampy.resample(data[:, 0], rates[0], 32000, filter='kaiser_best')
                soundfile.write(target_path_32, data_32, 32000)

        ratings_updated.append(SnodgrassWordRating(rating.word, rating.order, rating.p_score, rating.p_delay,
                                                   rating.duration, rating.synonym, rating.comment, rating.vp,
                                                   rating.date, target_path))
    with open(os.path.join(processed_data_dir, ratings_file), 'wb') as f:
        pickle.dump(ratings_updated, f, protocol=pickle.HIGHEST_PROTOCOL)


def ratings_stats(ratings_file):
    with open(os.path.join(processed_data_dir, ratings_file), 'rb') as f:
        ratings: List[SnodgrassWordRating] = pickle.load(f)

    bad_files = 0
    very_bad_files = 0
    p_scores = np.zeros(4, dtype=np.int32)
    for rating in ratings:
        if 'abgeschnitten' in rating.comment:
            bad_files += 1
        if rating.wav_path is None:
            very_bad_files += 1
        if not response_missing(rating):
            idx = rating.p_score - 1
            if idx > 3 or idx < 0:
                raise RuntimeError('Invalid rating: {0}'.format(idx))
            p_scores[idx] += 1

    print('Bad files {0}/{1}'.format(bad_files, len(ratings)))
    print('Very bad files {0}/{1}'.format(very_bad_files, len(ratings)))
    return p_scores


if __name__ == '__main__':
    def prepare_rating_data(scp_file_name, full=True):
        ratings_dir = collect_csv_ratings(scp_file_name, include_missing=full)
        update_ratings_with_fixed_audio(ratings_dir)


    # Uncomment to run manual rating data collection for this dataset
    # prepare_rating_data('all_snodgrass_cleaned_v5_test.scp')

    ratings_stats('all_snodgrass_cleaned_v5_test_ratings_full')
    # p_scores = ratings_stats('all_snodgrass_cleaned_v3_dev_ratings')
    # p_scores_train = ratings_stats('all_snodgrass_cleaned_v4_train_ratings_full')
    # p_scores += p_scores_train
    # print(p_scores)

    ## Noise removal notes:
    # TODO: noise removal with sox removes about 20-30 ms of audio at the end. why? how to fix? could not fix
    # noise removal:
    # for filename in $(ls -1 | grep _32.wav); do sox $filename ${filename%.*}-clean.wav noisered ~/noise.profile 0.2; done
    # actually better with parallel:
    # find . -name '*_32.wav' | parallel 'sox {} {.}-clean.wav noisered ~/noise.profile 0.2'

    # padding a bit at the end for now | this is for batch denoising with sox for normal usage, not just for VAD purposes
    # find . -name '*.wav' | parallel 'sox -v 0.90 {} -p remix 1,2 | sox -p {.}-denoised_at_0.2.wav noisered ~/noise.profile 0.2 pad 0 0.030'
