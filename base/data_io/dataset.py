import abc
import os
import pickle
import sys
from itertools import islice, cycle
from typing import Type

import numpy as np

from base import util
from base.common import key2word, key2dataset


class Dataset(metaclass=abc.ABCMeta):

    def __init__(self, data_path, parent_dataset_path=None, training=True, logger=None, variance_normalization=False,
                 noise_multiplier=0, noise_prob=1, mean_subtraction=False, supplement_rare_with_noisy=False,
                 supplement_seed=112):
        self.data_path = data_path
        self.word2idxs = {}
        self.idx2word = []
        self.idx2source_dataset = []
        self.idx2key = []
        self.data = []
        self.training = training
        self.noise_multiplier = noise_multiplier
        self.noise_prob = noise_prob

        util.warn_or_print(logger, 'Loading {0}, train = {1}'.format(data_path, training))
        if not training and parent_dataset_path is None:
            util.warn_or_print(logger, 'Non-training mode is selected, but parent_dataset_path is None')
            util.warn_or_print(logger, 'A non-training dataset must always have the parent specified, otherwise'
                                       'the data mean and other derived values will be incorrect. Aborting.')
            sys.exit(-1)

        for i, (key, mat) in enumerate(self._raw_data_iterator()):
            word = key2word(key)
            dataset = key2dataset(key)

            word_example_idx = self.word2idxs.get(word, [])
            word_example_idx.append(i)
            self.word2idxs[word] = word_example_idx
            self.idx2word.append(word)
            self.idx2source_dataset.append(dataset)
            self.idx2key.append(key)
            self.data.append(mat)

        self.idx2word = np.array(self.idx2word)
        self.idx2source_dataset = np.array(self.idx2source_dataset)
        self.idx2key = np.array(self.idx2key)
        for key in self.word2idxs:
            self.word2idxs[key] = np.array(self.word2idxs[key])
        self.counts = {key: self.word2idxs[key].shape[0] for key in self.word2idxs}

        if parent_dataset_path is None:
            self.mean = self._calculate_feature_mean()
            self.std = self._calculate_feature_std()
        else:
            self.load_derived_data(parent_dataset_path)

        if mean_subtraction:
            util.warn_or_print(logger, 'Applying mean subtraction')
            self.data = np.array([(segment - self.mean).astype(np.float32) for segment in self.data])
        elif variance_normalization:
            util.warn_or_print(logger, 'Applying variance normalization')
            self.data = np.array([((segment - self.mean) / self.std).astype(np.float32) for segment in self.data])
        else:
            util.warn_or_print(logger, 'No mean subtraction')
            self.data = np.array([segment.astype(np.float32) for segment in self.data])

        # TODO: sort keys before doing anything else? (for identical behaviour between Kaldi and LMDB-exported datasets)

        self.noisy = np.zeros(len(self.data), dtype=bool)
        if supplement_rare_with_noisy and training:
            state_before = np.random.get_state()
            np.random.seed(supplement_seed)

            before_mean_examples = int(np.ceil(np.mean(list(self.counts.values()))))
            util.warn_or_print(logger, 'Supplementing rare classes with noisy examples up to {0} total'.format(
                before_mean_examples))
            util.warn_or_print(logger, 'Original example count: {0}'.format(len(self.data)))
            for word, count in self.counts.items():
                if count < before_mean_examples:
                    to_add = before_mean_examples - count
                    augmented_source_idx = []
                    augmented_examples = []
                    for orig_idx in islice(cycle(self.word2idxs[word]), to_add):
                        orig_data = self.data[orig_idx]
                        augmented_source_idx.append(orig_idx)
                        augmented_examples.append(orig_data + noise_multiplier * np.random.normal(loc=0,
                                                                                                  scale=self.std,
                                                                                                  size=orig_data.shape))
                    augmented_source_idx = np.array(augmented_source_idx)
                    augmented_examples = np.array(augmented_examples)
                    self.data = np.concatenate((self.data, augmented_examples))
                    self.noisy = np.concatenate((self.noisy, np.ones(to_add, dtype=bool)))
                    self.word2idxs[word] = np.concatenate(
                        (self.word2idxs[word], np.arange(len(self.data) - to_add, len(self.data), dtype=np.int32)))
                    self.idx2word = np.concatenate((self.idx2word, [word for _ in augmented_examples]))
                    self.idx2key = np.concatenate((self.idx2key, [self.idx2key[x] for x in augmented_source_idx]))
                    self.idx2source_dataset = np.concatenate(
                        (self.idx2source_dataset, [self.idx2source_dataset[x] for x in augmented_source_idx]))

            self.counts = {key: self.word2idxs[key].shape[0] for key in self.word2idxs}
            util.warn_or_print(logger, 'Augmented example count: {0}'.format(len(self.data)))
            np.random.set_state(state_before)

        self.feature_dim = self.data[0].shape[1]
        self.mean_examples = int(np.ceil(np.mean(list(self.counts.values()))))

        # siamese training setup, ignoring words with 1 example
        self.siamese_words = np.array(sorted([key for key in self.word2idxs if self.counts[key] > 1]))
        self.num_siamese_words = self.siamese_words.shape[0]

        # classifier training setup
        self.all_words = np.array(sorted(list(self.counts.keys())))
        if parent_dataset_path is None:
            self.word2id = {key: i for i, key in enumerate(self.all_words)}
        else:
            new_words = np.array([x not in self.word2id for x in self.all_words])
            if np.any(new_words):
                max_given_id = max(self.word2id.values())
                for i, x in enumerate(self.all_words[new_words]):
                    self.word2id[x] = max_given_id + i + 1
        self.idx2word_id = np.array([self.word2id[word] for word in self.idx2word], dtype=np.int32)

    def _calculate_feature_mean(self):
        means = [np.mean(segment, axis=0) for segment in self.data]
        weights = [segment.shape[0] for segment in self.data]
        return np.average(means, weights=weights, axis=0)

    def _calculate_feature_std(self):
        per_segment_variance = np.array([np.mean((x - self.mean) ** 2, axis=0, dtype=np.float64) for x in self.data])
        return np.sqrt(np.average(per_segment_variance, weights=np.array([x.shape[0] for x in self.data]), axis=0))

    def _calculate_mean(self, data):
        means = [np.mean(segment, axis=0) for segment in data]
        weights = [segment.shape[0] for segment in data]
        return np.average(means, weights=weights, axis=0)

    def _calculate_std(self, data, mean):
        per_segment_variance = np.array([np.mean((x - mean) ** 2, axis=0, dtype=np.float64) for x in data])
        return np.sqrt(np.average(per_segment_variance, weights=np.array([x.shape[0] for x in data]), axis=0))

    def _zero_pad(self, indices, lengths, batch_first):
        # sorting by descending length for pytorch's pack_padded_sequence
        descending_length = lengths.argsort()[::-1]
        lengths[:] = lengths[descending_length]
        indices[:] = indices[descending_length]

        noise_multiplier = self.noise_multiplier if self.training and np.random.uniform() < self.noise_prob else 0

        if batch_first:
            padded = np.zeros((len(indices), lengths.max(), self.feature_dim), dtype=np.float32)
            for i, (x, l) in enumerate(zip(self.data[indices], lengths)):
                padded[i, :l, :] = x[:l]
                if noise_multiplier > 0 and not self.noisy[indices[i]]:
                    padded[i, :l, :] += noise_multiplier * np.random.normal(loc=0, scale=self.std, size=x[:l].shape)
        else:
            padded = np.zeros((lengths.max(), len(indices), self.feature_dim), dtype=np.float32)
            for i, (x, l) in enumerate(zip(self.data[indices], lengths)):
                padded[:l, i, :] = x[:l]
                if noise_multiplier > 0 and not self.noisy[indices[i]]:
                    padded[:l, i, :] += noise_multiplier * np.random.normal(loc=0, scale=self.std, size=x[:l].shape)

        orig_order = np.zeros_like(descending_length, dtype=np.int32)
        orig_order[descending_length] = np.arange(lengths.shape[0], dtype=np.int32)

        return padded, lengths, indices, orig_order

    def zero_padded_data(self, indices, batch_first=False):
        """Pad acoustic features to max sequence length"""

        lengths = np.array([v.shape[0] for v in self.data[indices]], dtype=np.int32)
        return self._zero_pad(indices, lengths, batch_first)

    def zero_padded_data_parts_augmented_online(self, indices, batch_first=False):
        """Pad acoustic features to max sequence length, augmented by taking equal-length parts
        (up to 20%, ..., 90% of the frame sequence) of the training examples for ~50% of the batches"""

        part_size = np.random.choice(np.linspace(0.2, 0.9, 8), 1)[0]
        if np.random.uniform() < 0.5:  # only do for 50% of the batches
            part_size = 1

        lengths = np.array([max(int(v.shape[0] * part_size), 2) for v in self.data[indices]], dtype=np.int32)
        return self._zero_pad(indices, lengths, batch_first)

    # Are deltas and delta-deltas interfering with this augmentation?
    def zero_padded_data_parts_augmented_offline(self, anchor_batch, same_batch, other_batch, batch_first=False):
        """Pad acoustic features to max sequence length, augmented by taking equal-length parts
        (up to 20%, ..., 90% of the frame sequence) of the training examples for ~30% of the batch"""

        mask = np.zeros(anchor_batch.shape[0], dtype=bool)
        mask[:int(mask.shape[0] / 3)] = True
        np.random.shuffle(mask)
        part_size = np.random.choice(np.linspace(0.2, 0.9, 8), 1)[0]
        min_frames = 20

        lengths_anchor = np.zeros_like(anchor_batch, dtype=np.int32)
        lengths_same = np.zeros_like(same_batch, dtype=np.int32)
        lengths_other = np.zeros_like(other_batch, dtype=np.int32)
        for i in range(anchor_batch.shape[0]):
            # if randomly selected, excluding examples that are too short
            # should reduce forcing equivalence between non-matching parts
            if mask[i] and self.data[anchor_batch[i]].shape[0] >= min_frames \
                    and self.data[same_batch[i]].shape[0] >= min_frames:
                lengths_anchor[i] = int(self.data[anchor_batch[i]].shape[0] * part_size)
                lengths_same[i] = int(self.data[same_batch[i]].shape[0] * part_size)
                # After some consideration I decided that the augmentation doesn't transfer so well to other words
                # words often ahve common parts, especially at the beginning, and especially in German
                for j in range(other_batch.shape[1]):
                    # lengths_other[i, j] = max(int(self.data[other_batch[i, j]].shape[0] * part_size), 2)
                    lengths_other[i, j] = self.data[other_batch[i, j]].shape[0]
            else:
                lengths_anchor[i] = self.data[anchor_batch[i]].shape[0]
                lengths_same[i] = self.data[same_batch[i]].shape[0]
                for j in range(other_batch.shape[1]):
                    lengths_other[i, j] = self.data[other_batch[i, j]].shape[0]

        indices = np.concatenate((anchor_batch, same_batch, other_batch.reshape(-1, order='F')))
        lengths = np.concatenate((lengths_anchor, lengths_same, lengths_other.reshape(-1, order='F')))
        return self._zero_pad(indices, lengths, batch_first)

    def classes(self, indices):
        return self.idx2word_id[indices]

    def siamese_triplet_epoch(self, batch_size, num_other=10, batch_first=False, augment_parts=False):
        """If self.training is True, the batches are of size (2 + num_other) * batch_size,
        where the first batch_size elements are the anchors, the second batch_size elements are the 'same class'
        examples, and each next batch_size elements are examples from a random other class.
         If self.training is False, each batch is simply batch_size samples, until the data runs out"""

        if self.training:
            epoch_length = self.mean_examples * self.num_siamese_words
            anchor_idx = np.zeros(epoch_length, dtype=np.int32)
            same_idx = np.zeros(epoch_length, dtype=np.int32)
            other_idx = np.zeros((epoch_length, num_other), dtype=np.int32)

            for i, word in enumerate(self.siamese_words):
                word_examples = self.word2idxs[word]
                with_replacement = self.counts[word] < self.mean_examples
                start = i * self.mean_examples
                end = (i + 1) * self.mean_examples
                anchor_idx[start:end] = np.random.choice(word_examples, size=self.mean_examples,
                                                         replace=with_replacement)
                same_idx[start:end] = np.random.choice(word_examples, size=self.mean_examples, replace=with_replacement)

                # fix pairs where the "same" word example happened to be the anchor example
                for idx in np.where(anchor_idx[start:end] == same_idx[start:end])[0]:
                    idx = idx + start
                    same_idx[idx] = np.random.choice(word_examples[word_examples != anchor_idx[idx]], 1)
                pass

                other_words = self.siamese_words[self.siamese_words != word]
                for other in range(num_other):
                    other_word = np.random.choice(other_words, 1)[0]
                    # if both the anchor and the other word have few examples, there will be a lot of duplicates,
                    # but hopefully taking the max of the loss over the num_other possibilities will reduce the
                    # duplicate count
                    other_word_examples = self.word2idxs[other_word]
                    other_idx[start:end, other] = np.random.choice(other_word_examples, size=self.mean_examples,
                                                                   replace=self.counts[other_word] < self.mean_examples)
                    other_words = other_words[other_words != other_word]  # prevent duplicate word selection

            idx = np.arange(epoch_length)
            np.random.shuffle(idx)
            anchor_idx = anchor_idx[idx]
            same_idx = same_idx[idx]
            other_idx = other_idx[idx]

            for batch_start in range(0, epoch_length, batch_size):
                anchor_batch = anchor_idx[batch_start:batch_start + batch_size]
                same_batch = same_idx[batch_start:batch_start + batch_size]
                other_batch = other_idx[batch_start:batch_start + batch_size]
                if augment_parts:
                    yield self.zero_padded_data_parts_augmented_offline(anchor_batch, same_batch, other_batch,
                                                                        batch_first=batch_first)
                else:
                    yield self.zero_padded_data(
                        np.concatenate((anchor_batch, same_batch, other_batch.reshape(-1, order='F'))),
                        batch_first=batch_first)

        else:
            # don't care about shuffling here
            for batch_start in range(0, self.data.shape[0], batch_size):
                yield self.zero_padded_data(np.arange(batch_start, min(batch_start + batch_size, self.data.shape[0])),
                                            batch_first=batch_first)

    # TODO: decouple sampler from dataset? | yes, definitely, and make augmentations easier configurable and extendable
    def siamese_margin_loss_epoch(self, batch_size, examples_per_word, batch_first=False, augment_parts=False):
        n_words = batch_size // examples_per_word

        if self.training:
            for _ in range(0, self.data.shape[0], batch_size):
                indices = np.zeros(batch_size, dtype=np.int32)
                words = np.random.choice(self.siamese_words, n_words, replace=False)
                for i, word in enumerate(words):
                    indices[i * examples_per_word:(i + 1) * examples_per_word] = \
                        np.random.choice(self.word2idxs[word], examples_per_word,
                                         replace=self.counts[word] < examples_per_word)
                if augment_parts:
                    yield self.zero_padded_data_parts_augmented_online(indices, batch_first=batch_first)
                else:
                    yield self.zero_padded_data(indices, batch_first=batch_first)

        else:
            for batch_start in range(0, self.data.shape[0], batch_size):
                yield self.zero_padded_data(np.arange(batch_start, min(batch_start + batch_size, self.data.shape[0])),
                                            batch_first=batch_first)

    def classifier_epoch(self, batch_size, batch_first=False):
        if self.training:
            # TODO: would something like torch.utils.data.WeightedRandomSampler be better here?
            # at least the torch.multinomial(self.weights, self.num_samples, self.replacement) part;
            # or just use a weight with the cross-entropy loss?
            # apparently that can be unstable, as we are dynamically changing the learning rate,
            # dependant on the batch composition (high lr if many examples have large weights, low if small weights)
            idx = np.zeros(self.mean_examples * self.all_words.shape[0], dtype=np.int32)
            for i, word in enumerate(self.all_words):
                with_replacement = self.counts[word] < self.mean_examples
                start = i * self.mean_examples
                end = (i + 1) * self.mean_examples
                word_examples = self.word2idxs[word]
                idx[start:end] = np.random.choice(word_examples, size=self.mean_examples, replace=with_replacement)

            np.random.shuffle(idx)
            for batch_start in range(0, self.data.shape[0], batch_size):
                yield self.zero_padded_data(idx[batch_start:batch_start + batch_size], batch_first=batch_first)
        else:
            # don't care about shuffling here
            for batch_start in range(0, self.data.shape[0], batch_size):
                yield self.zero_padded_data(np.arange(batch_start, min(batch_start + batch_size, self.data.shape[0])),
                                            batch_first=batch_first)

    def dump_derived_data(self):
        data_path_no_ext = os.path.splitext(self.data_path)[0]
        dump_mean_to = '{0}_mean'.format(data_path_no_ext)
        dump_std_to = '{0}_std'.format(data_path_no_ext)
        dump_word2id_to = '{0}_word2id'.format(data_path_no_ext)

        self.mean.dump(dump_mean_to)
        self.std.dump(dump_std_to)
        with open(dump_word2id_to, 'wb') as f:
            pickle.dump(self.word2id, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load_derived_data(self, source_data_path):
        scp_no_ext = os.path.splitext(source_data_path)[0]
        mean_file = '{0}_mean'.format(scp_no_ext)
        std_file = '{0}_std'.format(scp_no_ext)
        word2id_file = '{0}_word2id'.format(scp_no_ext)

        self.mean = np.load(mean_file, allow_pickle=True)
        self.std = np.load(std_file, allow_pickle=True)
        with open(word2id_file, 'rb') as f:
            self.word2id = pickle.load(f)

    @abc.abstractmethod
    def _raw_data_iterator(self):
        """Must produce (key, features) pairs."""
        return


def _print_patients(data_train, data_dev, data_test):
    from base.common import snodgrass_key2patient
    for ds in [data_train, data_dev, data_test]:
        patients = np.unique([snodgrass_key2patient(ds.idx2key[i]) for i in range(ds.data.shape[0]) if
                              ds.idx2source_dataset[i] == 'snodgrass'])
        print(patients)


def get_dataset_class_for_format(fmt, logger=None) -> Type[Dataset]:
    if fmt == 'scp':
        from base.data_io.kaldi_dataset import KaldiDataset
        util.warn_or_print(logger, 'Selecting KaldiDataset for data handling')
        return KaldiDataset
    elif fmt == 'lmdb':
        from base.data_io.lmdb_dataset import LMDBDataset
        util.warn_or_print(logger, 'Selecting LMDBDataset for data handling')
        return LMDBDataset
    else:
        msg = 'Unsupported data format: {0}'.format(fmt)
        util.warn_or_print(logger, msg)
        raise RuntimeError(msg)


def get_dataset_class_for_path(dataset_path, logger) -> Type[Dataset]:
    # get extension, remove leading dot, and pass as the format name
    return get_dataset_class_for_format(os.path.splitext(dataset_path)[1][1:], logger)
