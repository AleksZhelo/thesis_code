import time

import lmdb

from base.common import get_dataset_paths
from base.data_io.dataset import Dataset, _print_patients
from base.data_io.proto import tensor_pb2, utils
from conf import current_dataset


class LMDBDataset(Dataset):
    def __init__(self, data_path, parent_dataset_path=None, training=True, logger=None, variance_normalization=False,
                 noise_multiplier=0, noise_prob=1, mean_subtraction=False, supplement_rare_with_noisy=False,
                 supplement_seed=112):
        self.env = lmdb.open(data_path, max_readers=1, readonly=True, lock=False,
                             readahead=True, meminit=False)
        super(LMDBDataset, self).__init__(data_path, parent_dataset_path, training, logger, variance_normalization,
                                          noise_multiplier, noise_prob, mean_subtraction, supplement_rare_with_noisy,
                                          supplement_seed)

    def _raw_data_iterator(self):
        def generator():
            with self.env.begin(write=False) as txn:
                for key, serialized_str in txn.cursor():
                    tensor_protos = tensor_pb2.TensorProtos()
                    tensor_protos.ParseFromString(serialized_str)
                    features = utils.tensor_to_numpy_array(tensor_protos.protos[0])
                    yield key.decode('utf-8'), features

        return generator()


def __main():
    start = time.time()

    train_path, dev_path, test_path = get_dataset_paths(current_dataset, fmt='lmdb')
    data_train = LMDBDataset(train_path, noise_multiplier=1.0, noise_prob=0.5,
                             supplement_rare_with_noisy=False,
                             supplement_seed=112)
    data_dev = LMDBDataset(dev_path, parent_dataset_path=train_path, training=False)
    data_test = LMDBDataset(test_path, parent_dataset_path=train_path, training=False)

    _print_patients(data_train, data_dev, data_test)

    test = next(data_train.siamese_triplet_epoch(32, augment_parts=True))
    test = next(data_train.siamese_margin_loss_epoch(50, 5))

    print('LMDB: {0}'.format(time.time() - start))


if __name__ == '__main__':
    __main()
    # __main_snodgrass_test()
    # __main_external_test()
    # __main_independent_test()
    # __dump_numpy_txt()
