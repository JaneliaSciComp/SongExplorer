#a function that inputs the full path (including possibly a recording letter) to
#a file containing the audio recording, an interval of time, and some keyword
#arguments and returns the sampling rate, shape of entire recording (not just
#the interval), and requested data as int16.  if {start,stop}_tic are None, return
#the entire recording
def audio_read(fullpath, start_tic, stop_tic, **kw):

    # load data, determine sampling rate and length, and do any special processing

    return sampling_rate, nsamples_nchannels, slice_of_data

# a function that returns a list of file extensions which this plugin can handle
def audio_read_exts(**kw):
    return []  # e.g. ['.wav', '.WAV']

# a function that returns a dictionary that maps logical recordings to channels in the file
def audio_read_rec2ch(fullpath, **kw):
    return {}  # e.g. {'A':[0], 'B':[1]}, or {'A':[0,1]}

# a function that strips the recording suffix, if any
def audio_read_strip_rec(fullpath, **kw):
    return {}  # e.g. foo.wav-recA -> foo.wav

def audio_read_init(**kw):
    pass
