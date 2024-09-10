# .aq8 files from Electrical Penetration Graph (EGP; https://epgsystems.eu)
# systems can be read in directly.  the .D0x files that stylet+ automatically
# creates are not needed (nor are any .A0x files).

#audio_read_plugin="load-epg"
#audio_read_plugin_kwargs={"nchan":8, ncomments":3, "Fs":"smpl.frq= ([0-9.]+)Hz"}

import re
import numpy as np
import os

def audio_read(fullpath_aq8_rec, start_tic, stop_tic,
               nchan=8, ncomments=3, Fs="smpl.frq= ([0-9.]+)Hz", **kw):
    tmp = fullpath_aq8_rec.split('-')
    fullpath_aq8, rec = '-'.join(tmp[:-1]), tmp[-1]

    if not start_tic:  start_tic=0

    with open(fullpath_aq8, 'rb') as fid:
        for _ in range(ncomments):
            line = fid.readline().decode()
            m = re.search(Fs, line)
            if m:  sampling_rate = float(m.group(1))
        n0 = fid.tell()
        n1 = fid.seek(0,2)
        nsamples = (n1-n0)//4//nchan
        fid.seek(n0)
        if not stop_tic:  stop_tic=nsamples
        fid.seek(4*nchan*start_tic, 1)
        b = fid.read(4*nchan*(stop_tic-start_tic))

    v = np.frombuffer(b, dtype=np.float32)
    a = np.reshape(v, (-1,nchan))

    chs = audio_read_rec2ch()[rec]
    s = a[:, chs]

    c = (s / 10 * np.iinfo(np.int16).max).astype(np.int16)

    return sampling_rate, (nsamples,len(chs)), c

def audio_read_exts(nchan=8, **kw):
    return ['.aq'+str(nchan)]

def audio_read_rec2ch(nchan=8, **kw):
    return {"rec"+chr(65+i):[i] for i in range(nchan)}

def audio_read_init(**kw):
    pass
