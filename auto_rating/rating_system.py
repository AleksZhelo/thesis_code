import math
import os
import time
from collections import namedtuple
from typing import List

import numpy as np
import torch
from scipy.spatial.distance import cdist

from acoustic_word_embeddings.core.util.args_util import parse_auto_rating_args
from acoustic_word_embeddings.core.util.common import load_embeddings
from acoustic_word_embeddings.core.util.net_util import load_net
from acoustic_word_embeddings.gen_embeddings import get_or_generate_embeddings
from auto_rating.vad.segmentation import segment_generator, plain_audio_generator
from base import util
from base.common import response_missing, response_with_synonym
from base.sound_util import time2frames, frames2time
from base.util import load_pickled, save_pickled
from conf import processed_data_dir
from dataset_prep.core.common import scp2word_lengths_file
from dataset_prep.core.features import audio2lmfe, audio2lmfe_reverse
from dataset_prep.snodgrass import SnodgrassWordRating

NetAnnotatedSegment = namedtuple('NetAnnotatedSegment',
                                 ['start_sec', 'end_sec', 'segment_idx', 'dists', 'frame_means',
                                  'word', 'vp', 'date', 'source_path'])


def new_features(segment_audio, sample_rate, feature_mean, feature_std, mean_sub, var_norm, backwards=False):
    features_with_deltas = audio2lmfe(segment_audio, sample_rate) if not backwards else audio2lmfe_reverse(
        segment_audio, sample_rate)
    if mean_sub:
        features_with_deltas -= feature_mean
    elif var_norm:
        features_with_deltas = (features_with_deltas - feature_mean) / feature_std

    return features_with_deltas.reshape(features_with_deltas.shape[0], 1, -1).astype(np.float32)


def mean_dist_to_reference_examples(net_output, test_word, reference_vecs, word_idxs, metric):
    vecs = reference_vecs[word_idxs[test_word]]
    dists = []
    for x in net_output:
        dists_x = cdist(x[np.newaxis, :], vecs, metric=metric)
        dists.append(np.mean(dists_x))
    return np.array(dists)


def evaluate_stepwise(net, config, features_with_deltas, rating, reference_vecs, word_idxs):
    with torch.no_grad():
        data = torch.from_numpy(features_with_deltas)
        if config.model.use_cuda:
            data = data.cuda()

        net_output = net.stepwise_embeddings(data)
        net_output = net_output.detach().cpu().numpy()

    return mean_dist_to_reference_examples(net_output, rating.word, reference_vecs, word_idxs,
                                           net.loss_delegate.metric()), features_with_deltas


def evaluate_segment_stepwise(net, config, feature_mean, feature_std, rating, segment_audio, sample_rate,
                              reference_vecs, word_idxs, mean_sub, var_norm):
    features_with_deltas = new_features(segment_audio, sample_rate, feature_mean, feature_std, mean_sub, var_norm)

    return evaluate_stepwise(net, config, features_with_deltas, rating, reference_vecs,
                             word_idxs)


def evaluate_feature_vectors(net, config, data, lengths, rating, reference_vecs, word_idxs):
    with torch.no_grad():
        data = torch.from_numpy(data)
        lengths = torch.from_numpy(lengths)
        if config.model.use_cuda:
            data = data.cuda()
            lengths = lengths.cuda()

        net_output = net.forward((data, lengths))
        net_output = net_output.detach().cpu().numpy()

    return mean_dist_to_reference_examples(net_output, rating.word, reference_vecs, word_idxs,
                                           net.loss_delegate.metric())


def load_common_rating_data(ratings_file_or_data, run_dir, run_epoch):
    train_epoch_embeddings, _, _ = get_or_generate_embeddings(run_dir, run_epoch)
    words_train, datasets_train, vecs_train, counts_train, word_idxs_train = load_embeddings(
        train_epoch_embeddings[run_epoch])

    if isinstance(ratings_file_or_data, list):
        ratings = ratings_file_or_data
    else:
        ratings: List[SnodgrassWordRating] = load_pickled(os.path.join(processed_data_dir, ratings_file_or_data))
    return ratings, vecs_train, word_idxs_train


def net_annotate_vad_segmentation(run_dir, run_epoch, vad_aggressiveness, ratings_file, load_cleaned=True,
                                  skip_starting=0):
    net, config, _, _, _, _, train_scp, feature_mean, feature_std, word2id, mean_sub, var_norm = \
        load_net(run_dir, epoch=run_epoch, logger=None, train=False)
    ratings, vecs_train, word_idxs_train = load_common_rating_data(ratings_file, run_dir, run_epoch)

    util.ensure_exists('output')
    output_file = 'output/{0}_epoch_{1}_{2}_full_{3}_skip{4:.3f}.netrating' \
        .format(os.path.basename(run_dir), run_epoch, os.path.basename(ratings_file),
                '-noise_vad' if load_cleaned else 'raw_vad', skip_starting)

    # apparently [[]] * len(ratings) returns copies of *the same* empty list
    output: List[List[NetAnnotatedSegment]] = [[] for _ in range(len(ratings))]
    for rating, rating_idx, segment_idx, start_sec, end_sec, segment_audio, sample_rate, num_segments in \
            segment_generator(ratings, vad_aggressiveness, load_cleaned=load_cleaned, skip_starting=skip_starting):
        if segment_audio.shape[0] == 0:
            output[rating_idx].append(NetAnnotatedSegment(0, 0, 0, np.array([1000]), np.array([1000]), rating.word,
                                                          rating.vp, rating.date, rating.wav_path))
        else:
            dists, features = evaluate_segment_stepwise(net, config, feature_mean, feature_std, rating, segment_audio,
                                                        sample_rate, vecs_train, word_idxs_train, mean_sub, var_norm)

            output[rating_idx].append(
                NetAnnotatedSegment(start_sec, end_sec, segment_idx, dists, features[:, 0, :].mean(axis=1), rating.word,
                                    rating.vp, rating.date, rating.wav_path))
        if segment_idx == num_segments - 1:
            print('Finished rating {0}'.format(rating_idx))

    save_pickled(output, output_file)
    if hasattr(net, 'beta'):
        beta = net.beta.detach().cpu().numpy()
        with open("output/{0}_epoch_{1}.beta".format(os.path.basename(run_dir), run_epoch), 'wb') as f:
            np.save(f, beta)


def subsegment_starts(full_frame_count, segment_duration_frames, spacing_frames):
    starts = [i * spacing_frames for i in
              range(math.floor((full_frame_count - segment_duration_frames) / spacing_frames) + 1)]
    if len(starts) == 0:
        starts.append(0)
    return starts


def stack_features(features, num_segments, duration_frames, subsegment_portions, size_multiplier):
    # XXX: the idea was to evaluate distances to parts of the segment (say 25%, 50%, 75%, 100%)
    # and select the best segment based on the evolution of these distances
    # this was not useful, so only the 100% part is used for the decision now
    features_plus_extra = \
        np.zeros((duration_frames, num_segments * size_multiplier, features[0].shape[2]), dtype=np.float32)
    lengths = np.zeros(num_segments * size_multiplier, dtype=np.int32)

    for i in range(num_segments):
        features_plus_extra[:, i, :] = features[i][:, 0, :]
        lengths[i] = duration_frames
        for j in range(1, size_multiplier):
            subsegment_frames = int(subsegment_portions[j - 1] * duration_frames)
            idx = j * num_segments + i
            features_plus_extra[:subsegment_frames, idx, :] = features[i][:subsegment_frames, 0, :]
            lengths[idx] = subsegment_frames

    return features_plus_extra, lengths


def evaluate_stacked_features(net, config, features_plus_extra, lengths, rating, vecs_train, word_idxs_train,
                              num_segments, size_multiplier):
    dists = evaluate_feature_vectors(net, config, features_plus_extra, lengths, rating, vecs_train,
                                     word_idxs_train)
    dists = dists.reshape(num_segments, size_multiplier, order='F')
    return np.flip(dists, axis=1)


def select_best_segmentation(stacked_dists, starts, end_sec, max_length, skip_starting, starts_in_sec=False):
    bs_idx = np.argmin(stacked_dists[:, -1])
    best_start = (frames2time(starts[bs_idx]) if not starts_in_sec else starts[bs_idx]) + skip_starting
    max_duration = max_length + 0.5
    best_end = min(best_start + max_duration, end_sec)
    best_duration = best_end - best_start

    return bs_idx, best_start, best_end, best_duration


def net_annotate_sliding_window_framewise(run_dir, run_epoch, ratings_file_or_object, skip_starting=0,
                                          reference_vecs_override=None, reference_word_idxs_override=None,
                                          save=True, ratings_name=None, output_dir=None, plot_mode=False):
    time_start = time.time()

    if save:
        if ratings_name is None:
            ratings_name = os.path.basename(ratings_file_or_object)

        if output_dir is None:
            output_dir = 'output'
        util.ensure_exists(output_dir)

        output_file = '{0}_epoch_{1}_{2}_full{3}_skip{4:.3f}.netrating_faster' \
            .format(os.path.basename(run_dir), run_epoch, ratings_name,
                    'own_segmentation', skip_starting)
        output_file = os.path.join(output_dir, output_file)

    net, config, _, _, _, _, train_scp, feature_mean, feature_std, word2id, mean_sub, var_norm = \
        load_net(run_dir, epoch=run_epoch, logger=None, train=False)
    ratings, vecs_train, word_idxs_train = load_common_rating_data(ratings_file_or_object, run_dir, run_epoch)
    word_lengths = load_pickled(scp2word_lengths_file(train_scp))

    reference_vecs = reference_vecs_override if reference_vecs_override is not None else vecs_train
    reference_word_idxs = reference_word_idxs_override if reference_word_idxs_override is not None else word_idxs_train

    output: List[List[NetAnnotatedSegment]] = [[] for _ in range(len(ratings))]
    for rating, rating_idx, start_sec, end_sec, audio, sample_rate in \
            plain_audio_generator(ratings, skip_starting=skip_starting):
        if audio.shape[0] == 0:
            output[rating_idx].append(NetAnnotatedSegment(0, 0, 0, np.array([1000]), np.array([1000]), rating.word,
                                                          rating.vp, rating.date, rating.wav_path))
        else:
            mean_length, max_length = word_lengths[rating.word]
            spacing_frames = 5
            # TODO: half of mean duration may not be the best choice for every word
            duration_frames = time2frames(mean_length / 2)

            full_features = new_features(audio, sample_rate, feature_mean, feature_std, mean_sub, var_norm)
            starts = subsegment_starts(full_features.shape[0], duration_frames, spacing_frames)

            # much faster than segmenting first and then getting the features of each small segment
            features = [(full_features[s:s + duration_frames]) for s in starts]
            num_segments = len(features)

            subsegment_portions = [0.75, 0.5, 0.25]
            size_multiplier = len(subsegment_portions) + 1

            features_plus_extra, lengths = stack_features(features, num_segments, duration_frames, subsegment_portions,
                                                          size_multiplier)

            stacked_dists = evaluate_stacked_features(net, config, features_plus_extra, lengths, rating, reference_vecs,
                                                      reference_word_idxs, num_segments, size_multiplier)

            bs_idx, best_start, best_end, best_duration = \
                select_best_segmentation(stacked_dists, starts, end_sec, max_length, skip_starting)
            best_duration_frames = time2frames(best_duration)

            def plot_dists(savefig=False):
                import matplotlib.pyplot as plt
                from matplotlib import rc

                rc('text', usetex=True)
                rc('font', size=12)
                rc('legend', fontsize=12)
                font = {'family': 'serif', 'serif': ['cmr10']}
                rc('font', **font)

                if not response_missing(rating) and not response_with_synonym(rating):
                    p_delay_adjusted = rating.p_delay
                    plt.axvline(p_delay_adjusted, color='xkcd:bright lavender', dashes=[5, 5], zorder=2,
                                label='Word start')
                    plt.axvline(p_delay_adjusted + rating.duration, color='xkcd:light grass green', dashes=[1, 1],
                                zorder=2, label='Word end')

                plt.plot([frames2time(x) + skip_starting for x in starts], stacked_dists[:, -1], zorder=1,
                         color='xkcd:cobalt blue')
                plt.axvline(best_start, color='xkcd:lightish red', dashes=[1, 0], zorder=2, label='Word start guess')
                plt.xlabel('Time (s)')
                plt.ylabel('Avg distance to reference examples')
                plt.legend()
                if savefig:
                    plt.savefig('plots_output/recording_dists_{0:04}.pdf'.format(rating_idx), dpi=300,
                                bbox_inches='tight', pad_inches=0)
                else:
                    plt.show()
                plt.clf()

            if plot_mode:
                if not response_missing(rating) and not response_with_synonym(rating):
                    plot_dists(savefig=True)
                if rating_idx >= 10:
                    break

            dists_best_guess, features_best_guess = \
                evaluate_stepwise(net, config, full_features[starts[bs_idx]:starts[bs_idx] + best_duration_frames],
                                  rating, reference_vecs, reference_word_idxs)

            output[rating_idx].append(NetAnnotatedSegment(best_start, best_end, 0, dists_best_guess,
                                                          features_best_guess[:, 0, :].mean(axis=1), rating.word,
                                                          rating.vp, rating.date, rating.wav_path))
        print('Finished rating number {0}'.format(rating_idx + 1))

    if save and not plot_mode:
        save_pickled(output, output_file)
        if hasattr(net, 'beta'):
            beta = net.beta.detach().cpu().numpy()
            beta_out_file = os.path.join(output_dir, "{0}_epoch_{1}.beta".format(os.path.basename(run_dir), run_epoch))
            with open(beta_out_file, 'wb') as f:
                np.save(f, beta)

    print('Elapsed sec: {0:.3f}'.format(time.time() - time_start))
    return output, net.beta.detach().cpu().numpy() if hasattr(net, 'beta') else None


def __main():
    args = parse_auto_rating_args()
    run_dir = args.run_dir
    run_epoch = args.run_epoch
    ratings_file = args.ratings_file
    vad = args.vad

    if vad:
        print('Using VAD')
        net_annotate_vad_segmentation(run_dir=run_dir, run_epoch=run_epoch, vad_aggressiveness=3,
                                      ratings_file=ratings_file,
                                      load_cleaned=True, skip_starting=0)
    else:
        print('Using own segmentation approach')
        net_annotate_sliding_window_framewise(run_dir=run_dir, run_epoch=run_epoch,
                                              ratings_file_or_object=ratings_file,
                                              skip_starting=0.3,
                                              save=True)


if __name__ == '__main__':
    __main()
