import math
import os
import re
import string

from conf import processed_data_dir, res_dir


def get_dataset_paths(dataset):
    train_path = os.path.join(processed_data_dir, '{0}_train.scp'.format(dataset))
    dev_path = os.path.join(processed_data_dir, '{0}_dev.scp'.format(dataset))
    test_path = os.path.join(processed_data_dir, '{0}_test.scp'.format(dataset))

    return train_path, dev_path, test_path


def key2word(line):
    return line.split('_')[0]


def key2order(line):
    return int(line.split('_')[1]) - 1


def key2dataset(line):
    return line.split('_')[2]


def snodgrass_key2patient(line):
    # for other datasets this is the session
    return line.split('_')[3]


def snodgrass_key2date(line):
    # for other datasets this is the bundle, which usually has underscores too, so only the first part would be returned
    return line.split('_')[4]


def snodgrass_key2all(line):
    parts = line.split('_')
    word = parts[0]
    order = int(line.split('_')[1]) - 1
    dataset = parts(2)
    vp = parts[3]
    date = parts[4]
    return word, order, dataset, vp, date


def response_missing(rating):
    return math.isnan(rating.p_delay) or math.isnan(rating.duration)


__remove_punctuation_dict = {ord(c): None for c in string.punctuation}


def __clean_synonym_text(syn):
    cleaned = re.sub('\([^\(\)]+\)', '', syn)  # remove stuff in parentheses
    cleaned = re.sub('.+->', '', cleaned)  # remove anything before and including ->
    cleaned = re.sub('[dD]ialekt', '', cleaned)
    return cleaned.strip()


def response_with_synonym(rating):
    if rating.synonym is not None and rating.synonym.strip() != "":
        words = [x.translate(__remove_punctuation_dict).strip() for x in
                 re.split("\W+", __clean_synonym_text(rating.synonym))]
        return rating.word not in words
    else:
        return False


def load_snodgrass_words():
    with open(os.path.join(res_dir, 'snodgrass_words.txt'), 'r') as f:
        snodgrass_words = [line.rstrip('\n') for line in f]
    return snodgrass_words
