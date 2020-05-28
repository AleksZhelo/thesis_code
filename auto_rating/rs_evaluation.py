import pickle
from collections import namedtuple
from typing import List

import numpy as np

from auto_rating.rating_system import NetAnnotatedSegment
from base.common import response_missing, response_with_synonym
from base.sound_util import frames2time
from base.util import load_pickled
from dataset_prep.snodgrass import SnodgrassWordRating

Evaluation = namedtuple('Evaluation',
                        ['threshold_immediate', 'threshold_exhaustive', 'tpr', 'fpr', 'tp', 'fp', 'tn', 'fn',
                         'precision', 'recall', 'accuracy'])


def get_best_frame_with_rise(dists, max_dist_rise):
    best_frame = np.argmin(dists)
    for i, check in enumerate(dists[best_frame:] < dists[best_frame] + max_dist_rise):
        if not check:
            break
    frames_before_rise = i

    end_frame = best_frame + frames_before_rise
    time = frames2time(end_frame + 1)

    return best_frame, frames_before_rise, end_frame, time


def threshold_net_output(net_output_file, thresholds_immediate, thresholds_exhaustive,
                         max_dist_rise=0.001, min_frame_rise_len=None,
                         check_dists_for_end=True):
    # XXX: for sliding window segmentation the two-stage thresholding idea does not really apply,
    #  as there is only one potential detection at the end, and so only the higher threshold applies
    net_annotated_recordings: List[List[NetAnnotatedSegment]] = load_pickled(net_output_file)

    all_ratings = []
    for thr_immediate, thr_exhaustive in zip(thresholds_immediate, thresholds_exhaustive):
        net_ratings = [None] * len(net_annotated_recordings)
        for rec_idx, rec_segments in enumerate(net_annotated_recordings):
            for k, segment_rating in enumerate(rec_segments):

                dists = segment_rating.dists if isinstance(segment_rating.dists, np.ndarray) else np.array(
                    segment_rating.dists)
                start_sec = segment_rating.start_sec
                segment_idx = segment_rating.segment_idx

                best_frame, frames_before_rise, end_frame, time = \
                    get_best_frame_with_rise(dists, max_dist_rise)
                if not check_dists_for_end:  # when using the backwards-trained net the end is not determined here
                    time = segment_rating.end_sec - segment_rating.start_sec
                    frames_before_rise = -1

                if min_frame_rise_len is None or (frames_before_rise >= min_frame_rise_len):
                    if dists[best_frame] <= thr_immediate:
                        net_ratings[rec_idx] = (start_sec, time, dists[best_frame], segment_idx, frames_before_rise)
                        break
                    elif dists[best_frame] <= thr_exhaustive:
                        if net_ratings[rec_idx] is None:
                            net_ratings[rec_idx] = (start_sec, time, dists[best_frame], segment_idx, frames_before_rise)
                        elif net_ratings[rec_idx][2] > dists[best_frame]:
                            net_ratings[rec_idx] = (start_sec, time, dists[best_frame], segment_idx, frames_before_rise)

                if segment_idx == len(rec_segments) - 1 and net_ratings[rec_idx] is None:
                    net_ratings[rec_idx] = (None, None, None, len(rec_segments), 0)
        all_ratings.append((thr_immediate, thr_exhaustive, net_output_file, net_ratings))

    return all_ratings


def threshold_net_output_by_beta(net_output_file_or_data, beta_file_or_data, word2id_file, max_dist_rise=0.001,
                                 min_frame_rise_len=None, check_dists_for_end=True):
    if isinstance(net_output_file_or_data, list):
        net_annotated_recordings = net_output_file_or_data
        net_output_name = 'no_file'
    else:
        net_annotated_recordings: List[List[NetAnnotatedSegment]] = load_pickled(net_output_file_or_data)
        net_output_name = net_output_file_or_data

    if isinstance(beta_file_or_data, np.ndarray):
        beta = beta_file_or_data
    else:
        beta = np.load(beta_file_or_data)

    word2id = load_pickled(word2id_file)

    all_ratings = []
    net_ratings = [None] * len(net_annotated_recordings)
    for rec_idx, rec_segments in enumerate(net_annotated_recordings):
        for k, segment_rating in enumerate(rec_segments):

            dists = segment_rating.dists
            start_sec = segment_rating.start_sec
            segment_idx = segment_rating.segment_idx
            thr_immediate = beta[word2id[segment_rating.word]] - 0.20
            thr_exhaustive = beta[word2id[segment_rating.word]] + 0.20

            best_frame, frames_before_rise, end_frame, time = \
                get_best_frame_with_rise(dists, max_dist_rise)
            if not check_dists_for_end:  # when using the backwards-trained net the end is not determined here
                time = segment_rating.end_sec - segment_rating.start_sec
                frames_before_rise = -1

            if min_frame_rise_len is None or (frames_before_rise >= min_frame_rise_len):
                if dists[best_frame] <= thr_immediate:
                    net_ratings[rec_idx] = (start_sec, time, dists[best_frame], segment_idx, frames_before_rise)
                    break
                elif dists[best_frame] <= thr_exhaustive:
                    if net_ratings[rec_idx] is None:
                        net_ratings[rec_idx] = (start_sec, time, dists[best_frame], segment_idx, frames_before_rise)
                    elif net_ratings[rec_idx][2] > dists[best_frame]:
                        net_ratings[rec_idx] = (start_sec, time, dists[best_frame], segment_idx, frames_before_rise)

            if segment_idx == len(rec_segments) - 1 and net_ratings[rec_idx] is None:
                net_ratings[rec_idx] = (None, None, None, len(rec_segments), 0)
    all_ratings.append((-1, -1, net_output_name, net_ratings))

    return all_ratings


def compare_to_human(ratings, ratings_net, leeway_start=0.3, leeway_end=0.3):
    true_positive = 0
    false_positive_wrong_time = 0
    false_positive_wasnt_there = 0
    true_negative = 0
    false_negative = 0
    real_positive = 0
    real_negative = 0
    segment_counts = []
    distances = []
    frames_before_rise = []
    for i, (rating, rating_net) in enumerate(zip(ratings, ratings_net)):
        net_start = rating_net[0] if rating_net is not None else None
        net_duration = rating_net[1] if rating_net is not None else None
        net_dist = rating_net[2] if rating_net is not None else None
        net_n_segments = rating_net[3] + 1 if rating_net is not None else 0
        net_frames_before_rise = rating_net[4] + 1 if rating_net is not None and len(rating_net) > 4 else None

        segment_counts.append(net_n_segments)
        distances.append(net_dist)

        if not response_missing(rating) and not response_with_synonym(rating):
            real_positive += 1
        else:
            real_negative += 1

        if net_start is not None:
            if not response_missing(rating) and not response_with_synonym(rating):
                if abs(rating.p_delay - rating_net[0]) <= leeway_start and abs(
                        rating.p_delay + rating.duration - (net_start + net_duration)) <= leeway_end:
                    true_positive += 1
                    frames_before_rise.append((0, net_frames_before_rise))
                else:
                    false_positive_wrong_time += 1
                    frames_before_rise.append((1, net_frames_before_rise))
            else:
                false_positive_wasnt_there += 1
                frames_before_rise.append((2, net_frames_before_rise))
        else:
            if not response_missing(rating) and not response_with_synonym(rating):
                false_negative += 1
                frames_before_rise.append((4, net_frames_before_rise))
            else:
                true_negative += 1
                frames_before_rise.append((3, net_frames_before_rise))
    return false_negative, false_positive_wasnt_there, false_positive_wrong_time, real_negative, real_positive, \
           true_negative, true_positive, np.array(segment_counts), np.array(distances), np.array(frames_before_rise)


def evaluate_net_ratings_list(human_ratings_file_or_list, net_ratings_list, leeway_start=0.3, leeway_end=0.3,
                              verbose=False):
    if isinstance(human_ratings_file_or_list, list):
        ratings = human_ratings_file_or_list
    else:
        ratings: List[SnodgrassWordRating] = load_pickled(human_ratings_file_or_list)
    evaluations = []  # thr_immediate, threshold_exhaustive, TPR, FPR, tp, fp, tn, fn, precision, recall, accuracy

    for thr_immediate, thr_exhaustive, net_output_file, net_ratings in net_ratings_list:
        if verbose:
            print('{0} thresholded at {1} (immediate), {2} (exhaustive)'.format(net_output_file, thr_immediate,
                                                                                thr_exhaustive))

        false_negative, false_positive_wasnt_there, false_positive_wrong_time, real_negative, real_positive, \
        true_negative, true_positive, _, _, _ = compare_to_human(ratings, net_ratings, leeway_start=leeway_start,
                                                                 leeway_end=leeway_end)

        false_positive = false_positive_wrong_time + false_positive_wasnt_there
        total_good = true_positive + true_negative
        total_ratings = real_positive + real_negative

        net_positive = true_positive + false_positive
        if net_positive == 0:
            net_positive += 1
        result = Evaluation(thr_immediate, thr_exhaustive,
                            true_positive / (true_positive + false_negative),
                            false_positive / (false_positive + true_negative),
                            true_positive, false_positive, true_negative, false_negative,
                            true_positive / (net_positive),
                            true_positive / (true_positive + false_negative),
                            total_good / total_ratings * 100)

        evaluations.append(result)

        if verbose:
            print('Positive: true {0}, false wrong time {1}, false was not there: {2}'.format(true_positive,
                                                                                              false_positive_wrong_time,
                                                                                              false_positive_wasnt_there))
            print('Negative: true {0}, false {1}'.format(true_negative, false_negative))
            print('Real positive: {0}, real negative: {1}'.format(real_positive, real_negative))
            print(
                'Total correct: {0}/{1} ({2:.3f}%)'.format(total_good, total_ratings, total_good / total_ratings * 100))
            print('')

    return evaluations


def net_ratings_stats(human_ratings_file, net_ratings_list):
    with open(human_ratings_file, 'rb') as f:
        ratings: List[SnodgrassWordRating] = pickle.load(f)

    frames_before_rise = {}
    distances = {}
    segment_counts = {}

    for thr_immediate, thr_exhaustive, net_output_file, net_ratings in net_ratings_list:
        _, _, _, _, _, _, _, segment_counts_thr, distances_thr, frames_before_rise_thr = \
            compare_to_human(ratings, net_ratings)

        key = (thr_immediate, thr_exhaustive)
        frames_before_rise[key] = frames_before_rise_thr
        distances[key] = distances_thr
        segment_counts[key] = segment_counts_thr

    return frames_before_rise, distances, segment_counts
