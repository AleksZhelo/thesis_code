import csv
import datetime
import glob
import os

import numpy as np
from typing import List

import soundfile

from acoustic_word_embeddings.core.args_util import parse_rate_new_data_args
from auto_rating.rating_system import net_annotate_sliding_window_framewise
from auto_rating.rs_evaluation import threshold_net_output_by_beta
from base.common import load_snodgrass_words
from dataset_prep.snodgrass import SnodgrassWordRating


def snodgrass_audio2order(sound_paths):
    return list(map(lambda x: int(os.path.basename(x).split('-')[0][6:]) - 1, sound_paths))


def snodgrass_audio_chronological_sort_indices(sound_paths):
    return np.argsort(snodgrass_audio2order(sound_paths))


def gen_empty_ratings(folder):
    split = os.path.basename(folder).split('_')
    vp = split[-1]
    session_id = split[-2]
    wav_files = np.array(glob.glob(os.path.join(folder, '*.wav')))
    wav_files = wav_files[snodgrass_audio_chronological_sort_indices(wav_files)]
    orders = snodgrass_audio2order(wav_files)
    snodgrass_words = load_snodgrass_words()

    if len(wav_files) != 233:
        msg = 'Only {0} files in session folder {1} -- non-full sessions currently not supported for csv export'.format(
            len(wav_files), folder)
        print(msg)
        raise RuntimeError(msg)

    fake_ratings: List[SnodgrassWordRating] = []
    for wav_file, order in zip(wav_files, orders):
        word = snodgrass_words[order - 1]
        fake_ratings.append(SnodgrassWordRating(word, order, float('nan'), float('nan'), float('nan'), '', '',
                                                vp, session_id, wav_file))

    return fake_ratings


def rate_snodgrass_folder(folder, run_dir, run_epoch, word2id_file):
    fake_ratings = gen_empty_ratings(folder)

    annotations, beta = net_annotate_sliding_window_framewise(run_dir=run_dir, run_epoch=run_epoch,
                                                              ratings_file_or_object=fake_ratings,
                                                              skip_starting=0.3, save=False)

    # could use different thresholding here via threshold_net_output and a specific value
    thresholded_by_beta = threshold_net_output_by_beta(annotations, beta, word2id_file,
                                                       max_dist_rise=0.001, min_frame_rise_len=None)

    output_ratings = []
    for (start_sec, duration, dist, segment_idx, frames_before_rise), fake_rating in zip(thresholded_by_beta[0][3],
                                                                                         fake_ratings):
        output_ratings.append(
            SnodgrassWordRating(fake_rating.word, fake_rating.order, fake_rating.p_score, start_sec, duration,
                                fake_rating.synonym, fake_rating.comment, fake_rating.vp, fake_rating.date,
                                fake_rating.wav_path)
        )
    return output_ratings


def export_ratings_to_csv(file_path, ratings: List[SnodgrassWordRating], session_number, sessions_total):
    csv_file_header = ['Word', 'S Score', 'P Score', 'Old Scale', 'S Delay', 'P Delay', 'Duration',
                       'Synonym', 'Comment', 'Word signal time', 'Track length', 'VP code', 'Session ID',
                       'Order']
    with open(file_path, 'w', newline='', encoding="utf-8") as csv_file:
        csv_file.write('\ufeff')
        csv_writer = csv.writer(csv_file, quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(csv_file_header)

        for rating in ratings:
            track_length = round(soundfile.info(rating.wav_path).duration * 1000000)
            row = [rating.word, '', '', '', '', rating.p_delay, rating.duration, '', '', 0, track_length, rating.vp,
                   rating.date, rating.order * sessions_total + session_number]
            row = list(map(lambda x: x if x is not None else '', row))
            csv_writer.writerow(row)


def rate_patient_folder_for_multi_snodgrass(sessions_path, run_dir, run_epoch, word2id_file):
    sessions = [os.path.join(sessions_path, x) for x in os.listdir(sessions_path) if
                os.path.isdir(os.path.join(sessions_path, x))]

    rated_sessions = [rate_snodgrass_folder(session, run_dir, run_epoch, word2id_file) for session in sessions]
    # TODO: here we could refine the detected segments using VAD to remove excessive noise/silence at segment end,
    #  and reject segments containing only noise/silence. Since VAD needs a 32000 sampling rate, and works best for
    #  audio files somewhat cleaned from noise, doing this requires a lot of compute time. A simple implementation
    #  in rs_clean_with_vad.py did not work, needs testing and refinement.
    #  Steps to use:
    #  1. resample with resampy, see update_ratings_with_fixed_audio in collect_manual_rating_data.py
    #  2. noiseclean with sox: find . -name '*_32.wav' | parallel 'sox {} {.}-clean.wav noisered ~/noise.profile 0.2'
    #  3. Adapt the function "clean" from rs_clean_with_vad.py

    csv_files = []
    for i, (session_dir, net_ratings) in enumerate(zip(sessions, rated_sessions)):
        folder_name_parts = os.path.basename(session_dir).split('_')
        vp_code = folder_name_parts[-1]
        session_id = folder_name_parts[-2]

        csv_file = os.path.join(session_dir, '{0}_{1}_results-snodgrass.csv'.format(vp_code, session_id))
        export_ratings_to_csv(csv_file, net_ratings, i, len(sessions))
        csv_files.append(csv_file)

    meta_file = '{0}_auto_detected_on_{1}.nms'.format(
        os.path.basename(sessions_path), datetime.datetime.now().strftime("%d_%m_%Y--%H_%M_%S")
    )
    meta_file = os.path.join(sessions_path, meta_file)
    with open(meta_file, 'w', encoding='utf-8') as f:
        for csv_file in csv_files:
            f.write(os.path.relpath(csv_file, sessions_path))
            f.write(os.linesep)


def __test_single(args):
    # test_folder = '/home/aleks/work/BCI/data/nm_examples/2017-12-21_Soundfiles-Snodgrass_Koenig'
    test_folder = '/home/aleks/work/BCI/data/nm_examples/2017-02-23_Soundfiles-Snodgrass_mid_Eschbach'
    net_ratings = rate_snodgrass_folder(test_folder, args.run_dir, args.run_epoch, args.word2id_file)
    export_ratings_to_csv(os.path.join(test_folder, '11232_12323_results-snodgrass.csv'), net_ratings, 0, 1)


if __name__ == '__main__':
    args_ = parse_rate_new_data_args()
    # __test_single(args_)
    rate_patient_folder_for_multi_snodgrass(args_.patient_dir, args_.run_dir, args_.run_epoch, args_.word2id_file)

    # Example:
    # python rs_new_data.py
    # --patient_dir "/home/aleks/work/BCI/data/nm_examples/P10a Snodgrass"
    # --run_dir /home/aleks/projects/thesis/acoustic_word_embeddings/runs_cluster/siamese_53_20_12_2018
    # --run_epoch 66
    # --word2id_file /home/aleks/data/speech_processed/all_snodgrass_cleaned_v5_train_word2id
