from base import util


def reject_by_duration_sec(duration_sec, dataset, key, logger=None, verbose=True):
    if duration_sec < 0.200:
        if verbose:
            util.warn_or_print(logger,
                               'Skipping {0} - too short: {1:.2f} ms (< 200 ms)'.format(key, duration_sec * 1000))
        return True
    elif duration_sec > 1.5 and ('WaSeP' not in dataset):
        # WaSeP has some slowly pronounced words, not restricting length
        if verbose:
            util.warn_or_print(logger, 'Skipping {0} - too long: {1:.2f} sec (> 1.5 sec)'.format(key, duration_sec))
        return True
    else:
        return False


def reject_by_frame_count(features, word, dataset, key, logger=None, verbose=True):
    n_frames = features.shape[0]

    if len(word) <= 4 and n_frames < 20:
        if verbose:
            util.warn_or_print(logger, 'Skipping {0} - too few frames: {1} (< 20)'.format(key, n_frames))
        return True
    elif len(word) > 4 and n_frames < 25:
        if verbose:
            util.warn_or_print(logger, 'Skipping {0} - too few frames: {1} (< 25)'.format(key, n_frames))
        return True
    elif n_frames > 120 and len(word) < 12 and ('WaSeP' not in dataset):
        # WaSeP has some slowly pronounced words, not restricting length
        if verbose:
            util.warn_or_print(logger, 'Skipping {0} - too many frames: {1} (> 120)'.format(key, n_frames))
        return True
    elif n_frames > 140 and ('WaSeP' not in dataset):
        # WaSeP has some slowly pronounced words, not restricting length
        if verbose:
            util.warn_or_print(logger,
                               'Skipping {0} - too many frames: {1} (> 140) for a long word'.format(key, n_frames))
        return True
    else:
        return False


def reject_by_frame_count_snodgrass(features, word, dataset, key, logger=None, verbose=True):
    """Less strict than for other datasets, as the patients sometimes take a lot of time to pronounce a word"""
    n_frames = features.shape[0]

    if n_frames > 230:
        if verbose:
            util.warn_or_print(logger, 'Skipping {0} - too many frames: {1} (> 230) '.format(key, n_frames))
        return True
    else:
        return False
