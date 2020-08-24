import lmdb

import numpy as np

import base.data_io.proto.tensor_pb2 as tensor_pb2
import base.data_io.proto.utils as utils
from base.data_io.dataset import Dataset


def dataset2lmdb(dataset: Dataset, output_file):
    print(">>> Write database...")
    LMDB_MAP_SIZE = 1 << 40  # MODIFY
    print(LMDB_MAP_SIZE)
    env = lmdb.open(output_file, map_size=LMDB_MAP_SIZE)

    with env.begin(write=True) as txn:
        keys_sort_index = np.argsort(dataset.idx2key)
        keys = dataset.idx2key[keys_sort_index]
        data = dataset.data[keys_sort_index]

        for i, (key, features) in enumerate(zip(keys, data)):
            # Create TensorProtos
            tensor_protos = tensor_pb2.TensorProtos()
            features_tensor = utils.numpy_array_to_tensor(features)
            tensor_protos.protos.extend([features_tensor])

            txn.put(
                '{}'.format(key).encode('utf-8'),
                tensor_protos.SerializeToString()
            )

            if i % 16 == 0:
                print("Inserted {} rows".format(i))
