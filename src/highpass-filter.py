#audio_read_plugin="highpass-filter"
#audio_read_plugin_kwargs={"cutoff":1000, "order":2}

import numpy as np
import scipy.io.wavfile as spiowav
from scipy.signal import sosfiltfilt, butter

def audio_read(wav_path, start_tic, stop_tic, cutoff=1, order=2):
    sampling_rate, data = spiowav.read(wav_path, mmap=True)

    if np.ndim(data)==1:
        data = np.expand_dims(data, axis=1)
    if np.issubdtype(data.dtype, np.integer):
        data = data.astype('float32') / abs(np.iinfo(data.dtype).min)

    if not start_tic: start_tic=0
    if not stop_tic: stop_tic=np.shape(data)[0]+1

    start_tic_clamped = max(0, start_tic)
    stop_tic_clamped = min(np.shape(data)[0]+1, stop_tic)

    sos = butter(order, cutoff, 'highpass', fs=sampling_rate, output='sos')

    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.sosfiltfilt.html
    padlen = 3 * (2 * len(sos) + 1 - min((sos[:, 2] == 0).sum(), (sos[:, 5] == 0).sum()))
    padlenL = min(padlen, start_tic_clamped)
    padlenR = min(padlen, np.shape(data)[0] - stop_tic_clamped)

    data_filtered = sosfiltfilt(sos.astype('float32'),
                                data[start_tic_clamped-padlenL : stop_tic_clamped+padlenR],
                                axis=0,
                                padtype=None)
    data_unpadded = data_filtered[padlenL:-padlenR or None, :]

    return sampling_rate, data.shape, data_unpadded
