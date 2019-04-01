import numpy as np
import webrtcvad

from auto_rating.vad.webrtc_vad import frame_generator, vad_collector
from base.sound_util import read_wav_plus_resampled, time2sample, read_wav, sample2time


def segment_generator(ratings, vad_aggressiveness, load_cleaned=True, combine=True, skip_starting=0):
    vad = webrtcvad.Vad(vad_aggressiveness)

    for r_idx, rating in enumerate(ratings):
        # too slow
        # audio, sample_rate, bytes_for_vad, vad_rate = \
        #     read_wav_with_resampling(rating.wav_path, skip_starting=skip_starting)
        audio, sample_rate, bytes_for_vad, vad_rate = read_wav_plus_resampled(rating.wav_path,
                                                                              load_cleaned=load_cleaned,
                                                                              skip_starting=skip_starting)
        frames = frame_generator(10, bytes_for_vad, vad_rate)
        segment_gen = vad_collector(vad_rate, 10, 100, vad, list(frames))

        segments = []
        for _, start_sec, end_sec in segment_gen:
            segments.append((start_sec, end_sec,
                             audio[time2sample(start_sec, sample_rate): time2sample(end_sec, sample_rate)]))
        segments_augmented = [(start, end, np.copy(data)) for start, end, data in segments]

        if combine:
            # combining up to three consecutive segments
            for i in range(len(segments)):
                next_two = segments[i + 1:i + 3]
                start = segments[i][0]
                total_end = segments[i][1]
                for start_sec, end_sec, data in next_two:
                    if start_sec - total_end <= 1:  # up to one second gap
                        total_end = end_sec
                        segments_augmented.append(
                            (start, total_end,
                             audio[time2sample(start, sample_rate): time2sample(total_end, sample_rate)]))
                    else:
                        break

        if len(segments_augmented) > 0:
            for k, (start_sec, end_sec, segment_audio) in enumerate(segments_augmented):
                if end_sec - start_sec > 4:  # limiting max segment length to 4 seconds
                    end_sec = start_sec + 4
                    segment_audio = segment_audio[:time2sample(4, sample_rate)]
                yield rating, r_idx, k, start_sec + skip_starting, end_sec + skip_starting, segment_audio, sample_rate, \
                      len(segments_augmented)
        else:
            yield rating, r_idx, 0, 0, 0, np.empty(0), sample_rate, 0


def plain_audio_generator(ratings, skip_starting=0):
    for r_idx, rating in enumerate(ratings):
        audio, sample_rate = read_wav(rating.wav_path, skip_starting=skip_starting)

        start_sec = 0
        end_sec = sample2time(audio.shape[0], sample_rate)
        yield rating, r_idx, start_sec + skip_starting, end_sec + skip_starting, audio, sample_rate
