from cachetools import LRUCache

from base import sound_util


class WavCache(LRUCache):
    def __init__(self, maxsize, channel=0):
        def missing(key):
            """Retrieve the wav file"""
            data, rate = sound_util.read_wav(key, channel=channel)
            return rate, data

        def size_of(value):
            return value[1].nbytes

        super(WavCache, self).__init__(maxsize=maxsize, missing=missing, getsizeof=size_of)
