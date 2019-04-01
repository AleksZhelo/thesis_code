import errno
import logging
import os
import pickle
import sys


def ensure_exists(dir_path):
    """Credit to http://stackoverflow.com/a/5032238"""
    try:
        os.makedirs(dir_path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█'):
    """
    Slightly modified from https://stackoverflow.com/a/34325723

    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end='\r')
    # Print New Line on Complete
    if iteration == total:
        print()


def checkpoint_path2epoch(checkpoint):
    return int(os.path.splitext(checkpoint)[0].split('_')[-1])


def create_logger(name, log_file, level=logging.INFO):
    logger = logging.getLogger(name)
    formatter = logging.Formatter('[%(levelname)-8s] (%(processName)-11s) %(asctime)s %(message)s',
                                  "%H:%M:%S")
    file_handler = logging.FileHandler(log_file, 'w')
    file_handler.setFormatter(formatter)
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stdout_handler)
    logger.setLevel(level)
    return logger


def warn_or_print(logger, msg):
    if logger is None:
        print(msg)
    else:
        logger.warning(msg)


def critical_or_print(logger, msg):
    if logger is None:
        print(msg)
    else:
        logger.critical(msg)


def remove_all(a, b):
    return [x for x in a if x not in b]


def sum_dicts(x, y):
    if x is None:
        return y
    elif y is None:
        return x
    else:
        return {k: x.get(k, 0) + y.get(k, 0) for k in set(x) | set(y)}


def collapse_nested_dict(x):
    try:
        key = next(iter(x))
        return sum_dicts(x[key], collapse_nested_dict({k: v for k, v in x.items() if k != key}))
    except StopIteration:
        return x


def overlap(a, b):
    """
    Taken from https://stackoverflow.com/a/52388579

    Return the amount of overlap, in bp
    between a and b.
    If >0, the number of bp of overlap
    If 0,  they are book-ended.
    If <0, the distance in bp between them
    """

    return min(a[1], b[1]) - max(a[0], b[0])


def load_pickled(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def save_pickled(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
