from typing import List

import webrtcvad

from auto_rating.rating_system import NetAnnotatedSegment
from auto_rating.vad.webrtc_vad import frame_generator, vad_collector
from base.sound_util import read_wav_plus_resampled_segment, frames2time
from base.util import load_pickled, save_pickled


def clean(net_annotations_file, name, vad_aggressiveness=3, fix_no_voice=True, load_cleaned=True):
    vad = webrtcvad.Vad(vad_aggressiveness)

    net_annotated_recordings: List[List[NetAnnotatedSegment]] = load_pickled(net_annotations_file)

    net_annotated_recordings_filtered: List[List[NetAnnotatedSegment]] = []
    for rec_idx, rec_segments in enumerate(net_annotated_recordings):
        new_rec_annotations = []

        for k, segment_rating in enumerate(rec_segments):
            audio, sample_rate, bytes_for_vad, vad_rate = read_wav_plus_resampled_segment(segment_rating.source_path,
                                                                                          segment_rating.start_sec,
                                                                                          segment_rating.end_sec,
                                                                                          load_cleaned=load_cleaned)
            frames = frame_generator(10, bytes_for_vad, vad_rate)
            segment_gen = vad_collector(vad_rate, 10, 100, vad, list(frames))

            segments = []
            for _, start_sec, end_sec in segment_gen:
                segments.append((start_sec, end_sec))

            dists = segment_rating.dists
            workaround_frame_means = segment_rating.frame_means
            if len(segments) > 0:
                voice_end = segments[0][1]
                for i in range(workaround_frame_means.shape[0]):
                    workaround_frame_means[i] = 1 if frames2time(i + 1) <= voice_end else -1000
            else:
                if fix_no_voice:
                    dists[:] = 1.5  # invalidate dists if no voice detected

            new_rec_annotations.append(
                NetAnnotatedSegment(segment_rating.start_sec, segment_rating.end_sec, segment_rating.segment_idx,
                                    dists, workaround_frame_means,
                                    segment_rating.word, segment_rating.vp, segment_rating.date,
                                    segment_rating.source_path))

        net_annotated_recordings_filtered.append(new_rec_annotations)

    output_name = '{name}_vad{0}_bycleaned{1}_novoicefix{2}.netrating'.format(vad_aggressiveness,
                                                                              1 if load_cleaned else 0,
                                                                              1 if fix_no_voice else 0,
                                                                              name=name)
    save_pickled(net_annotated_recordings_filtered, output_name)


def __main():
    path = '/home/aleks/projects/thesis/auto_rating/output/siamese_53_20_12_2018_epoch_66_all_snodgrass_cleaned_v5_' \
           'test_ratings_full_fullown_segmentation_skip0.300.netrating_faster'

    clean(path, 'augparts-test', vad_aggressiveness=2, load_cleaned=True, fix_no_voice=False)


if __name__ == '__main__':
    # TODO: eventually did not work, re-check and improve
    __main()
