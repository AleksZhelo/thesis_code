import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile
from scipy import stats
from scipy.stats import pearsonr
from sklearn import metrics

# noinspection PyUnresolvedReferences
from auto_rating.rs_evaluation import net_ratings_stats, threshold_net_output, evaluate_net_ratings_list, \
    threshold_net_output_by_beta, get_best_frame_with_rise
from base import util, sound_util
from base.common import response_missing, response_with_synonym
from base.sound_util import time2sample
from base.util import overlap, load_pickled
from conf import processed_data_dir
from dataset_prep.snodgrass import SnodgrassWordRating


def plot_roc_curve(tpr, fpr, ax, label='', color='b'):
    roc_auc = metrics.auc(fpr, tpr)

    ax.plot(fpr, tpr, color=color, label='{0} AUC = {1:.2f}'.format(label, roc_auc))
    ax.legend(loc='lower right')
    ax.plot([0, 1], [0, 1], 'r--')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_ylabel('True Positive Rate')
    ax.set_xlabel('False Positive Rate')


def evaluations2df(data_two_stage):
    if not isinstance(data_two_stage, np.ndarray):
        data_two_stage = np.array(data_two_stage)

    data = {
        'thr': data_two_stage[:, 0],
        'thr_exh': data_two_stage[:, 1],
        'TP': data_two_stage[:, 4].astype(np.int32),
        'FP': data_two_stage[:, 5].astype(np.int32),
        'TN': data_two_stage[:, 6].astype(np.int32),
        'FN': data_two_stage[:, 7].astype(np.int32),
        'precision': data_two_stage[:, 8],
        'recall': data_two_stage[:, 9],
        'F1': stats.hmean(np.ma.masked_equal(data_two_stage[:, 8:10], 0), axis=1),
        'acc': data_two_stage[:, 10],
    }
    frame_two_stage = pd.DataFrame(data)
    return frame_two_stage


def get_comparison_roc_plot(evaluations_two_stage, evaluations_exhaustive, roc_title):
    data_two_stage = np.array(evaluations_two_stage)
    data_exhaustive = np.array(evaluations_exhaustive)

    # ROC plot
    f_roc, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))

    ax.set_title(roc_title)
    plot_roc_curve(data_two_stage[:, 2], data_two_stage[:, 3], ax, label='Two-stage')
    plot_roc_curve(data_exhaustive[:, 2], data_exhaustive[:, 3], ax, label='Exhaustive', color='g')

    return f_roc


def get_comparison_seg_histograms(human_ratings_file, net_ratings_list, title_list):
    _, _, segment_counts = net_ratings_stats(human_ratings_file, net_ratings_list)

    f_seg, ax = plt.subplots(nrows=1, ncols=len(net_ratings_list), figsize=(10, 5), sharey=True)

    for i, (thr_immediate, thr_exhaustive, _, net_ratings) in enumerate(net_ratings_list):
        seg_counts_at_thr = segment_counts[(thr_immediate, thr_exhaustive)]
        ax[i].hist(seg_counts_at_thr[np.logical_and(seg_counts_at_thr <= 30, seg_counts_at_thr > 0)], bins=30)
        ax[i].set_title(title_list[i])

    return f_seg


def overlap_with_human_rating(human_rating, net_start, net_duration):
    overlap_value = overlap((human_rating.p_delay, human_rating.p_delay + human_rating.duration),
                            (net_start, net_start + net_duration))

    return overlap_value


def correlate_to_human_scores(human_ratings_file, ratings_net, leeway_start=0.3, leeway_end=0.3):
    ratings: List[SnodgrassWordRating] = load_pickled(human_ratings_file)

    dists = []
    human_scores = []
    for i, (rating, rating_net) in enumerate(zip(ratings, ratings_net)):
        net_start = rating_net[0] if rating_net is not None else None
        net_duration = rating_net[1] if rating_net is not None else None
        net_dist = rating_net[2] if rating_net is not None else None
        net_n_segments = rating_net[3] + 1 if rating_net is not None else 0
        net_frames_before_rise = rating_net[4] + 1 if rating_net is not None and len(rating_net) > 4 else None

        if net_start is not None:
            if not response_missing(rating) and not response_with_synonym(rating):
                if abs(rating.p_delay - rating_net[0]) <= leeway_start and abs(
                        rating.p_delay + rating.duration - (net_start + net_duration)) <= leeway_end:
                    dists.append(net_dist)
                    human_scores.append(rating.p_score)
                else:
                    pass
            else:
                pass

    dists = np.array(dists)
    human_scores = np.array(human_scores)
    return pearsonr(1 / dists, human_scores)


def __main(net_output_file, beta_file, human_ratings_file, leeway_start=0.3, leeway_end=0.3):
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    pd.options.display.float_format = '{:,.2f}'.format

    is_rerated = 'rerated' in os.path.basename(net_output_file)
    word2id_file = os.path.join(processed_data_dir, 'all_snodgrass_cleaned_v5_train_word2id')
    beta_file = os.path.join('output', beta_file)

    # thr_imm = np.hstack(([0.0, 0.1], [0.2] * 10))
    # thr_exh = np.hstack(([0.1], np.linspace(0, 1, 11) + 0.2))
    thr_imm = np.hstack(([0.1, 0.2, 0.3], [0.4] * 11))
    thr_exh = np.hstack(([0.2, 0.3], np.linspace(0, 1.1, 12) + 0.4))
    rise = 0.001  # XXX: the idea was to take the segment at which the avg. distance has risen this much
    # from the minimum, but it appears that just the minimum works better
    thresholded_two_stage = threshold_net_output(net_output_file, thr_imm, thr_exh,
                                                 max_dist_rise=rise, min_frame_rise_len=None,
                                                 check_dists_for_end=not is_rerated)

    # thr_exh = np.linspace(0.1, 1, 10)
    thresholded_exhaustive = threshold_net_output(net_output_file, np.zeros_like(thr_exh), thr_exh,
                                                  max_dist_rise=rise, min_frame_rise_len=None,
                                                  check_dists_for_end=not is_rerated)

    thresholded_by_beta = threshold_net_output_by_beta(net_output_file, beta_file, word2id_file,
                                                       max_dist_rise=rise, min_frame_rise_len=None,
                                                       check_dists_for_end=not is_rerated)

    evaluations_two_stage = evaluate_net_ratings_list(human_ratings_file, thresholded_two_stage,
                                                      leeway_start=leeway_start,
                                                      leeway_end=leeway_end, verbose=False)
    # evaluations_exhaustive = evaluate_net_ratings_list(human_ratings_file, thresholded_exhaustive, leeway_start=leeway_start,
    #                                                  leeway_end=leeway_end, verbose=False)
    evaluations_beta = evaluate_net_ratings_list(human_ratings_file, thresholded_by_beta, leeway_start=leeway_start,
                                                 leeway_end=leeway_end, verbose=False)

    r, p_val = correlate_to_human_scores(human_ratings_file, thresholded_by_beta[0][-1], leeway_start=leeway_start,
                                         leeway_end=leeway_end)
    print('Thresholded by beta: r to P scores: {0:.3f}, p-value: {1:.3f}'.format(r, p_val))

    frame_two_stage = evaluations2df(evaluations_two_stage)
    # frame_exhaustive = evaluations2df(evaluations_exhaustive).drop('thr', 1)
    frame_by_beta = evaluations2df(evaluations_beta)
    print(frame_two_stage.to_latex(index=False))
    # print(frame_exhaustive.to_latex(index=False))
    print(frame_by_beta.to_latex(index=False))


def print_synonym_stats(ratings):
    synonyms = 0
    util.ensure_exists('test')
    for i, rating in enumerate(ratings):
        if not response_missing(rating) and response_with_synonym(rating):
            synonyms += 1
            print("{i:04d}: {0} -> {1}".format(rating.word, rating.synonym, i=i))
            data, rate = soundfile.read(rating.wav_path, always_2d=1)
            segment = data[time2sample(rating.p_delay, rate):time2sample(rating.p_delay + rating.duration, rate), 0]
            sound_util.write_array_to_wav(os.path.join('test', "{i:04d}".format(i=i)), segment, rate)
    print(synonyms, synonyms / len(ratings))


def print_p_delay_stats(ratings):
    p_delays = []
    for rating in ratings:
        if not response_missing(rating):
            if rating.p_delay > 50:
                key = '{0}_{1}_snodgrass_{2}_{3}'.format(rating.word, rating.order, rating.vp, rating.date)
                print(key, rating.p_delay, rating.duration, rating.comment)
            else:
                p_delays.append(rating.p_delay)

    p_delays = np.array(p_delays)

    print("min={min:.3f}, median={med:.3f}, mean={mean:.3f}, max={max:.3f}".format(min=p_delays.min(),
                                                                                   med=np.median(p_delays),
                                                                                   mean=p_delays.mean(),
                                                                                   max=p_delays.max()))


def __rate():
    net_output_file = 'siamese_53_20_12_2018_epoch_66_all_snodgrass_cleaned_v5_test_ratings_full_fullown_segmentation_skip0.300.netrating_faster'

    net_output_file = os.path.join('output', net_output_file)

    beta_file = 'siamese_53_20_12_2018_epoch_66.beta'

    human_ratings_file = os.path.join(processed_data_dir, 'all_snodgrass_cleaned_v5_test_ratings_full')

    __main(net_output_file, beta_file, human_ratings_file, leeway_start=0.3, leeway_end=0.3)
    __main(net_output_file, beta_file, human_ratings_file, leeway_start=0.2, leeway_end=0.2)


if __name__ == '__main__':
    __rate()
