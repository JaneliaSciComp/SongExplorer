# audio_read_plugin="load-wav"
# audio_read_plugin_kwargs={'mmap':True}

import numpy as np
import scipy.io.wavfile as spiowav

def audio_read(wav_path, start_tic, stop_tic, mmap=True):
    sampling_rate, data = spiowav.read(wav_path, mmap=mmap)

    if np.ndim(data)==1:
        data = np.expand_dims(data, axis=1)

    if not start_tic: start_tic=0
    if not stop_tic: stop_tic=np.shape(data)[0]+1

    start_tic_clamped = max(0, start_tic)
    stop_tic_clamped = min(np.shape(data)[0]+1, stop_tic)

    data_sliced = data[start_tic_clamped : stop_tic_clamped, :]

    return sampling_rate, data.shape, data_sliced

def audio_read_exts(**kw):
    return ['.wav', '.WAV']

def audio_read_rec2ch(wav_path, **kw):
    return {'recA':[0]}

def audio_read_strip_rec(wav_path, **kw):
    return wav_path

def audio_read_init(**kw):
    pass
