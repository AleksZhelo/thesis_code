import glob
import os

import numpy as np
import scipy.io.wavfile as wav

from conf import snodgrass_data_dir


def fix_snodgrass_sound(wav_path):
    rate, data = wav.read(wav_path)
    data.setflags(write=1)
    data[np.where(data == 0)] = 127
    wav.write(wav_path, rate, data)


def fix_all_snodgrass():
    """Fix popping sounds in old Snodgrass recordings."""
    wav_files = glob.glob(os.path.join(snodgrass_data_dir, '*', '*.wav'))
    for file in wav_files:
        fix_snodgrass_sound(file)


if __name__ == '__main__':
    fix_all_snodgrass()
