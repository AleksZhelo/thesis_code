import glob
import os
from typing import List

import numpy as np
import pandas as pd
from scipy import stats

from auto_rating.rs_analysis import evaluations2df
from auto_rating.rs_evaluation import threshold_net_output_by_beta, evaluate_net_ratings_list
from base.util import load_pickled
from conf import processed_data_dir
from dataset_prep.snodgrass import SnodgrassWordRating


def analyse_net_output(net_output_file, beta_file, word2id_file, human_ratings_list, leeway_start=0.3, leeway_end=0.3,
                       verbose=False):
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    pd.options.display.float_format = '{:,.2f}'.format

    rise = 0.001

    thresholded_by_beta = threshold_net_output_by_beta(net_output_file, beta_file, word2id_file,
                                                       max_dist_rise=rise, min_frame_rise_len=None,
                                                       check_dists_for_end=True)

    evaluations_beta = evaluate_net_ratings_list(human_ratings_list, thresholded_by_beta, leeway_start=leeway_start,
                                                 leeway_end=leeway_end, verbose=False)

    frame_by_beta = evaluations2df(evaluations_beta)
    if verbose:
        print(frame_by_beta.to_latex(index=False))
    return evaluations_beta[0].tp, evaluations_beta[0].fp, evaluations_beta[0].fn, evaluations_beta[0].tn


def evaluate_folds(fold_output_files, ratings_patient, all_sessions, beta_file, word2id_file,
                   leeway_start, leeway_end):
    results = []
    for file in fold_output_files:
        for session in all_sessions:
            if session in file:  # this session was excluded
                fold_ratings = [r for r in ratings_patient if r.date != session]

                res = analyse_net_output(file, beta_file, word2id_file, fold_ratings,
                                         leeway_start=leeway_start, leeway_end=leeway_end)
                results.append(res)

    # TP, FP, FN, TN
    total_results = np.array([np.array(x) for x in results]).sum(axis=0)
    precision = total_results[0] / (total_results[0] + total_results[1])
    recall = total_results[0] / (total_results[0] + total_results[2])
    accuracy = (total_results[0] + total_results[-1]) / (np.sum(total_results))
    print(
        'Precision: {0:.3f}, recall: {1:.3f}, F1: {2:.3f}, accuracy: {3:.3f}'.format(precision, recall,
                                                                                     stats.hmean([precision, recall]),
                                                                                     accuracy * 100)
    )


def __main():
    ratings_file = '/home/aleks/data/speech_processed/all_snodgrass_cleaned_v3_dev_ratings_full'
    patient = 'Eschbach'
    input_dir = 'patient_adaptation_test_output_Eschbach'
    run_prefix = 'siamese_43_20_12_2018_epoch_51_all_snodgrass_cleaned_v3_dev_ratings_full_patient_Eschbach'
    beta_file = [x for x in glob.glob(os.path.join(input_dir, '*.beta')) if
                 run_prefix.startswith(os.path.basename(os.path.splitext(x)[0]))][0]
    word2id_file = os.path.join(processed_data_dir, 'all_snodgrass_cleaned_v5_train_word2id')

    all_ratings: List[SnodgrassWordRating] = load_pickled(os.path.join(processed_data_dir, ratings_file))
    ratings_patient = [r for r in all_ratings if r.vp == patient]

    all_sessions = np.unique([r.date for r in ratings_patient])

    input_files = glob.glob(os.path.join(input_dir, '{0}*'.format(run_prefix)))
    non_adapted_runs = []
    only_new_adaptation = []
    add_new_adaptation = []
    average_with_new_adaptation = []

    for file in input_files:
        basename = os.path.basename(file)
        if 'adaptation' not in basename:
            non_adapted_runs.append(file)
        else:
            if 'only_new_session' in basename:
                only_new_adaptation.append(file)
            elif 'add_new_session' in basename:
                add_new_adaptation.append(file)
            elif 'average_with_new_session' in basename:
                average_with_new_adaptation.append(file)
            else:
                raise RuntimeError('Unexpected input file: {0}'.format(file))

    leeway_start = 0.3
    leeway_end = 0.3

    print('Leeway 0.3')
    print('Non-adapted')
    evaluate_folds(non_adapted_runs, ratings_patient, all_sessions, beta_file, word2id_file,
                   leeway_start=leeway_start, leeway_end=leeway_end)
    print('Only new')
    evaluate_folds(only_new_adaptation, ratings_patient, all_sessions, beta_file, word2id_file,
                   leeway_start=leeway_start, leeway_end=leeway_end)
    print('Add new')
    evaluate_folds(add_new_adaptation, ratings_patient, all_sessions, beta_file, word2id_file,
                   leeway_start=leeway_start, leeway_end=leeway_end)
    print('Average with new')
    evaluate_folds(average_with_new_adaptation, ratings_patient, all_sessions, beta_file, word2id_file,
                   leeway_start=leeway_start, leeway_end=leeway_end)

    print('\nLeeway 0.2')
    leeway_start = 0.2
    leeway_end = 0.2

    print('Non-adapted')
    evaluate_folds(non_adapted_runs, ratings_patient, all_sessions, beta_file, word2id_file,
                   leeway_start=leeway_start, leeway_end=leeway_end)
    print('Only new')
    evaluate_folds(only_new_adaptation, ratings_patient, all_sessions, beta_file, word2id_file,
                   leeway_start=leeway_start, leeway_end=leeway_end)
    print('Add new')
    evaluate_folds(add_new_adaptation, ratings_patient, all_sessions, beta_file, word2id_file,
                   leeway_start=leeway_start, leeway_end=leeway_end)
    print('Average with new')
    evaluate_folds(average_with_new_adaptation, ratings_patient, all_sessions, beta_file, word2id_file,
                   leeway_start=leeway_start, leeway_end=leeway_end)


if __name__ == '__main__':
    __main()
