import os
from itertools import islice, cycle

from typing import List, Dict

import numpy as np

from acoustic_word_embeddings.core.args_util import parse_patient_adaptation_test_args
from acoustic_word_embeddings.core.common import load_embeddings
from acoustic_word_embeddings.gen_embeddings import get_or_generate_embeddings
from auto_rating.rating_system import net_annotate_sliding_window_framewise
from base import util
from base.common import key2dataset, snodgrass_key2date, key2word
from base.util import load_pickled
from conf import processed_data_dir
from dataset_prep.snodgrass import SnodgrassWordRating


def adaptation_only_session_examples_if_available(vecs_train, word_idxs_train: Dict, session_vecs,
                                                  session_word_idxs: Dict):
    vecs_out = []
    words_out = []

    for word in set(word_idxs_train.keys()).union(set(session_word_idxs.keys())):
        train_idxs = word_idxs_train.get(word, [])
        session_idxs = session_word_idxs.get(word, [])

        if len(session_idxs) > 0:
            vecs_out.extend(session_vecs[session_idxs])
            words_out.extend([word] * len(session_idxs))
        elif len(train_idxs) > 0:
            vecs_out.extend(vecs_train[train_idxs])
            words_out.extend([word] * len(train_idxs))

    vecs_out = np.array(vecs_out)
    words_out = np.array(words_out)
    word_idxs_out = {key: np.where(words_out == key)[0] for key in np.unique(words_out)}

    return vecs_out, word_idxs_out


def adaptation_add_session_examples(vecs_train, word_idxs_train: Dict, session_vecs, session_word_idxs: Dict):
    vecs_out = []
    words_out = []

    for word in set(word_idxs_train.keys()).union(set(session_word_idxs.keys())):
        train_idxs = word_idxs_train.get(word, [])
        session_idxs = session_word_idxs.get(word, [])

        if len(session_idxs) > 0:
            vecs_out.extend(session_vecs[session_idxs])
            words_out.extend([word] * len(session_idxs))
        if len(train_idxs) > 0:
            vecs_out.extend(vecs_train[train_idxs])
            words_out.extend([word] * len(train_idxs))

    vecs_out = np.array(vecs_out)
    words_out = np.array(words_out)
    word_idxs_out = {key: np.where(words_out == key)[0] for key in np.unique(words_out)}

    return vecs_out, word_idxs_out


def adaptation_average_with_session_examples(vecs_train, word_idxs_train: Dict, session_vecs, session_word_idxs: Dict):
    """Relies on the distance calculation code averaging the distances over all reference examples for word, repeating
    the session examples increases their weight to half of the total"""
    vecs_out = []
    words_out = []

    for word in set(word_idxs_train.keys()).union(set(session_word_idxs.keys())):
        train_idxs = word_idxs_train.get(word, [])
        session_idxs = session_word_idxs.get(word, [])

        if len(session_idxs) > 0:
            if len(train_idxs) > len(session_idxs):
                idxs_to_add = list(islice(cycle(session_idxs), len(train_idxs)))  # repeat until same count as train
            else:
                idxs_to_add = session_idxs
            vecs_out.extend(session_vecs[idxs_to_add])
            words_out.extend([word] * len(idxs_to_add))
        if len(train_idxs) > 0:
            vecs_out.extend(vecs_train[train_idxs])
            words_out.extend([word] * len(train_idxs))

    vecs_out = np.array(vecs_out)
    words_out = np.array(words_out)
    word_idxs_out = {key: np.where(words_out == key)[0] for key in np.unique(words_out)}

    return vecs_out, word_idxs_out


def collect_session_embeddings_data(session, vecs_dev, keys_dev):
    sessions_vecs = []
    session_keys = []

    for i, key in enumerate(keys_dev):
        if key2dataset(key) == 'snodgrass' and snodgrass_key2date(key) == session:
            sessions_vecs.append(vecs_dev[i])
            session_keys.append(key)

    sessions_vecs = np.array(sessions_vecs)
    session_keys = np.array(session_keys)
    session_words = np.array([key2word(key) for key in session_keys])
    session_word_idxs = {key: np.where(session_words == key)[0] for key in np.unique(session_words)}

    return sessions_vecs, session_word_idxs


def patient_adaptation_test_on_dev(args):
    run_dir = args.run_dir
    run_epoch = args.run_epoch
    ratings_file = args.ratings_file
    patient = args.patient

    out_dir = 'patient_adaptation_test_output_{0}'.format(patient)
    util.ensure_exists(out_dir)

    if '_dev_' not in ratings_file:
        raise RuntimeError('Only ratings available in the dev dataset currently supported')

    train_epoch_embeddings, dev_epoch_embeddings, _ = \
        get_or_generate_embeddings(run_dir, run_epoch, dev_needed=True, test_needed=False)
    words_train, datasets_train, vecs_train, counts_train, word_idxs_train = load_embeddings(
        train_epoch_embeddings[run_epoch])
    words_dev, datasets_dev, vecs_dev, counts_dev, word_idxs_dev, keys_dev = load_embeddings(
        dev_epoch_embeddings[run_epoch], return_keys=True)

    all_ratings: List[SnodgrassWordRating] = load_pickled(os.path.join(processed_data_dir, ratings_file))
    ratings_patient = [r for r in all_ratings if r.vp == patient]

    all_sessions = np.unique([r.date for r in ratings_patient])
    print('{0} sessions for patient {1}'.format(len(all_sessions), patient))

    adaptation_functions = {
        'only_new_session': adaptation_only_session_examples_if_available,
        'add_new_session': adaptation_add_session_examples,
        'average_with_new_session': adaptation_average_with_session_examples
    }

    for session in all_sessions:
        fold_ratings = [r for r in ratings_patient if r.date != session]

        sessions_vecs, session_word_idxs = collect_session_embeddings_data(session, vecs_dev, keys_dev)

        ratings_name = '{0}_patient_{1}_except_{2}'.format(os.path.basename(ratings_file), patient, session)
        net_annotate_sliding_window_framewise(run_dir=run_dir,
                                              run_epoch=run_epoch,
                                              ratings_file_or_object=fold_ratings,
                                              skip_starting=0.3,
                                              save=True,
                                              ratings_name=ratings_name,
                                              output_dir=out_dir)

        for adaptation_type, method in adaptation_functions.items():
            reference_vecs, reference_word_idxs = method(vecs_train, word_idxs_train, sessions_vecs,
                                                         session_word_idxs)

            ratings_name = '{0}_patient_{1}_except_{2}_adaptation_{3}'.format(os.path.basename(ratings_file), patient,
                                                                              session, adaptation_type)
            net_annotate_sliding_window_framewise(run_dir=run_dir,
                                                  run_epoch=run_epoch,
                                                  ratings_file_or_object=fold_ratings,
                                                  skip_starting=0.3,
                                                  reference_vecs_override=reference_vecs,
                                                  reference_word_idxs_override=reference_word_idxs,
                                                  save=True,
                                                  ratings_name=ratings_name,
                                                  output_dir=out_dir)


def __main():
    args = parse_patient_adaptation_test_args()
    patient_adaptation_test_on_dev(args)


if __name__ == '__main__':
    __main()
