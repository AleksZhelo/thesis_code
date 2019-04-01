import contextlib
import math
import os
import wave

import numpy as np
import resampy
import soundfile
from scipy.io import wavfile as wav


def float2pcm(sig, dtype='int16'):
    """
    Source: https://github.com/mgeier/python-audio/blob/master/audio-files/utility.py
    Convert floating point signal with a range from -1 to 1 to PCM.
    Any signal values outside the interval [-1.0, 1.0) are clipped.
    No dithering is used.
    Note that there are different possibilities for scaling floating
    point numbers to PCM numbers, this function implements just one of
    them.  For an overview of alternatives see
    http://blog.bjornroche.com/2009/12/int-float-int-its-jungle-out-there.html
    Parameters
    ----------
    sig : array_like
        Input array, must have floating point type.
    dtype : data type, optional
        Desired (integer) data type.
    Returns
    -------
    numpy.ndarray
        Integer data, scaled and clipped to the range of the given
        *dtype*.
    See Also
    --------
    pcm2float, dtype
    """
    sig = np.asarray(sig)
    if sig.dtype.kind != 'f':
        raise TypeError("'sig' must be a float array")
    dtype = np.dtype(dtype)
    if dtype.kind not in 'iu':
        raise TypeError("'dtype' must be an integer type")

    i = np.iinfo(dtype)
    abs_max = 2 ** (i.bits - 1)
    offset = i.min + abs_max
    return (sig * abs_max + offset).clip(i.min, i.max).astype(dtype)


def time2sample(time_sec, sample_rate):
    return int(time_sec * sample_rate)


def sample2time(sample, sample_rate):
    return sample / sample_rate


def time2frames(time):
    return math.ceil((time - 0.015) / 0.010)


def frames2time(frames):
    return 0.010 * frames + (0.015 if frames > 0 else 0)


def read_wav(path, channel=0, skip_starting=0):
    data, rate = soundfile.read(path, always_2d=True)
    data = data[:, channel]

    if skip_starting > 0:
        data = data[int(rate * skip_starting):]

    return data, rate


def read_wav_with_resampling(path, channel=0, skip_starting=0):
    data, rate = soundfile.read(path, always_2d=True)
    data = data[:, channel]
    if skip_starting > 0:
        data = data[int(rate * skip_starting):]

    rate_32 = 32000
    data_32 = resampy.resample(data, rate, rate_32, filter='kaiser_best')

    return data, rate, float2pcm(data_32).tobytes(), rate_32


def read_wav_plus_resampled(path, channel=0, load_cleaned=True, skip_starting=0):
    data, rate = soundfile.read(path, always_2d=True)
    data = data[:, channel]
    if not load_cleaned:
        data_32, rate_32 = soundfile.read(os.path.splitext(path)[0] + '_32.wav', always_2d=True)
    else:
        data_32, rate_32 = soundfile.read(os.path.splitext(path)[0] + '_32-clean.wav', always_2d=True)

    if skip_starting > 0:
        data = data[int(rate * skip_starting):]
        data_32 = data_32[int(rate_32 * skip_starting):]

    return data, rate, float2pcm(data_32[:, channel]).tobytes(), rate_32


def read_wav_plus_resampled_segment(path, start_sec, end_sec, channel=0, load_cleaned=True):
    data, rate = soundfile.read(path, always_2d=True)
    data = data[:, channel]
    if not load_cleaned:
        data_32, rate_32 = soundfile.read(os.path.splitext(path)[0] + '_32.wav', always_2d=True)
    else:
        data_32, rate_32 = soundfile.read(os.path.splitext(path)[0] + '_32-clean.wav', always_2d=True)

    data = data[time2sample(start_sec, rate):time2sample(end_sec, rate)]
    data_32 = data_32[time2sample(start_sec, rate_32):time2sample(end_sec, rate_32)]

    return data, rate, float2pcm(data_32[:, channel]).tobytes(), rate_32


def write_array_to_wav(name, data, rate):
    wav.write('{0}.wav'.format(name), rate, data)


def write_pcm_wave(path, audio, sample_rate):
    with contextlib.closing(wave.open(path, 'wb')) as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio)
