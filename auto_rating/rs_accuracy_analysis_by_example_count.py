import os
from collections import namedtuple, Counter

from typing import List

import numpy as np

from auto_rating.rating_system import NetAnnotatedSegment
from auto_rating.rs_evaluation import threshold_net_output_by_beta
from base.common import response_missing, response_with_synonym
from base.util import load_pickled
from conf import processed_data_dir

NetRatedWord = namedtuple('NetRatedWord',
                          ['word', 'correct', 'vp', 'date', 'source_path'])


def compare_to_human_correctness(ratings, ratings_net, leeway_start=0.3, leeway_end=0.3):
    net_words = []
    for i, (rating, rating_net) in enumerate(zip(ratings, ratings_net)):
        net_start = rating_net[0] if rating_net is not None else None
        net_duration = rating_net[1] if rating_net is not None else None

        correct = False
        if net_start is not None:
            if not response_missing(rating) and not response_with_synonym(rating):
                if abs(rating.p_delay - rating_net[0]) <= leeway_start and abs(
                        rating.p_delay + rating.duration - (net_start + net_duration)) <= leeway_end:
                    correct = True
                else:
                    pass
            else:
                pass
        else:
            if not response_missing(rating) and not response_with_synonym(rating):
                pass
            else:
                correct = True

        net_words.append(NetRatedWord(rating.word, correct, rating.vp, rating.date, rating.wav_path))

    return net_words


def word2character_count_group(word):
    if len(word) <= 3:
        group = 0
    elif len(word) == 4:
        group = 1
    elif len(word) == 5:
        group = 2
    elif len(word) <= 7:
        group = 3
    elif len(word) <= 10:
        group = 4
    else:
        group = 5

    return group


def group_analysis(net_rated_words, grouping, groups_num):
    group_counts = np.zeros(groups_num)
    for x in net_rated_words:
        group_counts[grouping(x.word)] += 1
    print(group_counts, np.sum(group_counts))

    correct_per_group = np.zeros(groups_num)
    incorrect_per_group = np.zeros(groups_num)

    for x in net_rated_words:
        group_id = grouping(x.word)
        if x.correct:
            correct_per_group[group_id] += 1
        else:
            incorrect_per_group[group_id] += 1

    print(np.round(correct_per_group / (correct_per_group + incorrect_per_group) * 100, 3))


def __main():
    def word2example_count_group(word):
        examples = example_counts[word]
        if examples <= 5:
            group = 0
        elif examples <= 10:
            group = 1
        elif examples <= 25:
            group = 2
        elif examples <= 50:
            group = 3
        elif examples <= 100:
            group = 4
        else:
            group = 5

        return group

    net_output_file = '/home/aleks/projects/thesis/auto_rating/augparts-test_vad3_bycleaned1_novoicefix0.netrating'

    beta_file = 'output/siamese_53_20_12_2018_epoch_66.beta'

    human_ratings_file = os.path.join(processed_data_dir, 'all_snodgrass_cleaned_v5_test_ratings_full')

    word2id_file = os.path.join(processed_data_dir, 'all_snodgrass_cleaned_v5_train_word2id')

    word2id = load_pickled(word2id_file)
    example_counts = load_pickled('train_counts.pckl')

    thresholded_by_beta = threshold_net_output_by_beta(net_output_file, beta_file, word2id_file,
                                                       max_dist_rise=0.001,
                                                       min_frame_rise_len=None,
                                                       check_dists_for_end=True)

    net_rated_words = compare_to_human_correctness(load_pickled(human_ratings_file),
                                                   thresholded_by_beta[0][-1],
                                                   leeway_start=0.2,
                                                   leeway_end=0.2)

    counts = Counter([x.word for x in net_rated_words])

    correct = np.zeros(len(word2id))
    incorrect = np.zeros(len(word2id))

    for x in net_rated_words:
        if x.correct:
            correct[word2id[x.word]] += 1
        else:
            incorrect[word2id[x.word]] += 1

    # for word in counts:
    #     accuracy = correct[word2id[word]] / (correct[word2id[word]] + incorrect[word2id[word]]) * 100
    #     print('{0}: {1:.3f}% accuracy'.format(word, accuracy))

    group_analysis(net_rated_words, word2character_count_group, 6)
    group_analysis(net_rated_words, word2example_count_group, 6)





if __name__ == '__main__':
    __main()
