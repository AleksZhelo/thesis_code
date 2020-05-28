import math

import numpy as np
import speechpy
from python_speech_features import logfbank, delta, mfcc


def audio2lmfe_speechpy(audio, rate):
    # DONE: something weird with frame_stride? for a segment exactly 50 ms long gives only two frames,
    # although presumably three should fit: 0-25, 10-35, 20-45 ms?
    # apparently the framing code is somewhat bugged - it should have been like above
    fft_length = math.ceil(0.025 * rate)
    features = speechpy.feature.lmfe(audio - np.mean(audio), rate, frame_length=0.025, frame_stride=0.010,
                                     num_filters=40, low_frequency=20, high_frequency=7800, fft_length=fft_length)

    features_with_deltas = speechpy.feature.extract_derivative_feature(features)
    features_with_deltas = features_with_deltas.reshape(features_with_deltas.shape[0], -1, order='F')

    return features_with_deltas


def audio2lmfe(audio, rate):
    fft_length = math.ceil(0.025 * rate)  # TODO: should have been a power of 2 to enable FFT
    # pre-emphasis seems to be useful, Hamming windowing - not really
    features = logfbank(audio - np.mean(audio),
                        samplerate=rate, winlen=0.025, winstep=0.01, nfilt=40, nfft=fft_length, lowfreq=20,
                        highfreq=7800, preemph=0.97)

    deltas = delta(features, 2)
    delta_deltas = delta(deltas, 2)
    features_with_deltas = np.concatenate((features, deltas, delta_deltas), axis=1)

    return features_with_deltas


def audio2lmfe_reverse(audio, rate):
    return audio2lmfe(audio[::-1], rate)


def audio2mfcc(audio, rate):
    # Didn't work at all

    fft_length = math.ceil(0.025 * rate)
    features = mfcc(audio - np.mean(audio), samplerate=rate, winlen=0.025, winstep=0.01, numcep=13, nfilt=26,
                    nfft=fft_length, lowfreq=20, highfreq=7800, preemph=0.97, appendEnergy=True)

    deltas = delta(features, 2)
    delta_deltas = delta(deltas, 2)
    features_with_deltas = np.concatenate((features, deltas, delta_deltas), axis=1)

    return features_with_deltas
