# .aq8 files from Electrical Penetration Graph (EGP; https://epgsystems.eu)
# systems can be read in directly.  the .D0x files that stylet+ automatically
# creates are not needed (nor are any .A0x files).

#audio_read_plugin="load-epg"
#audio_read_plugin_kwargs={"nchan":8, ncomments":3, "Fs":"smpl.frq= ([0-9.,]+)Hz"}

import re
import numpy as np
import os
import scipy.io.wavfile as spiowav

def audio_read(fullpath_aq8_rec_or_wav, start_tic, stop_tic,
               nchan=8, ncomments=3, Fs="smpl.frq= +([0-9.,]+)Hz", mmap=True, **kw):
    if not start_tic:  start_tic=0
    ext = os.path.splitext(fullpath_aq8_rec_or_wav)[1]

    if ext.startswith(".aq"+str(nchan)):
        tmp = fullpath_aq8_rec_or_wav.split('-')
        fullpath_aq8, rec = '-'.join(tmp[:-1]), tmp[-1]
        with open(fullpath_aq8, 'rb') as fid:
            for _ in range(ncomments):
                line = fid.readline().decode()
                m = re.search(Fs, line)
                if m:  sampling_rate = round(float(m.group(1).replace(",", ".")))
            n0 = fid.tell()
            n1 = fid.seek(0,2)
            nsamples = (n1-n0)//4//nchan
            fid.seek(n0)
            if not stop_tic:  stop_tic=nsamples
            fid.seek(4*nchan*start_tic, 1)
            b = fid.read(4*nchan*(stop_tic-start_tic))

        v = np.frombuffer(b, dtype=np.float32)
        a = np.reshape(v, (-1,nchan))

        chs = audio_read_rec2ch(fullpath_aq8)[rec]
        s = a[:, chs]

        c = (s / 10 * np.iinfo(np.int16).max).astype(np.int16)

        return sampling_rate, (nsamples,len(chs)), c

    elif ext in ['.wav', '.WAV']:
        sampling_rate, data = spiowav.read(fullpath_aq8_rec_or_wav, mmap=mmap)

        if np.ndim(data)==1:
            data = np.expand_dims(data, axis=1)

        if not stop_tic: stop_tic=np.shape(data)[0]+1

        start_tic_clamped = max(0, start_tic)
        stop_tic_clamped = min(np.shape(data)[0]+1, stop_tic)

        data_sliced = data[start_tic_clamped : stop_tic_clamped, :]

        return sampling_rate, data.shape, data_sliced

    elif ext in Dexts:
        with open(fullpath_aq8_rec_or_wav, 'rb') as fid:
            for _ in range(ncomments):
                line = fid.readline().decode()
                m = re.search(Fs, line)
                if m:  sampling_rate = round(float(m.group(1).replace(",", ".")))
            n0 = fid.tell()
            n1 = fid.seek(0,2)
            nsamples = (n1-n0)//4
            fid.seek(n0)
            if not stop_tic:  stop_tic=nsamples
            fid.seek(4*start_tic, 1)
            b = fid.read(4*(stop_tic-start_tic))

        v = np.frombuffer(b, dtype=np.float32)
        a = np.reshape(v, (-1,1))
        c = (a / 10 * np.iinfo(np.int16).max).astype(np.int16)

        return sampling_rate, (nsamples,1), c

def audio_read_exts(nchan=8, **kw):
    return ['.aq'+str(nchan), '.wav', '.WAV', *Dexts]

def audio_read_rec2ch(fullpath_aq8_or_wav, nchan=8, **kw):
    ext = os.path.splitext(fullpath_aq8_or_wav)[1]
    if ext == ".aq"+str(nchan):
        return {"rec"+chr(65+i):[i] for i in range(nchan)}
    elif ext in ['.wav', '.WAV'] or ext in Dexts:
        return {'recA':[0]}

def audio_read_strip_rec(fullpath_aq8_rec_or_wav, nchan=8, **kw):
    ext = os.path.splitext(fullpath_aq8_rec_or_wav)[1]
    if ext.startswith(".aq"+str(nchan)):
        return fullpath_aq8_rec_or_wav[:-5]
    elif ext in ['.wav', '.WAV'] or ext in Dexts:
        return fullpath_aq8_rec_or_wav

def audio_read_init(**kw):
    global Dexts
    Dexts = ['.D'+str(x).zfill(2) for x in range(1,99)]
    pass
