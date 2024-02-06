# a function that inputs the full path to a file containing the audio recording,
# an interval of time, and some keyword arguments and returns the sampling
# rate, shape of entire recording (not just the interval), and requested data as int16
def audio_read(fullpath, start_tic, stop_tic, **kw):

    # load data, determine sampling rate and length, and do any special processing

    return sampling_rate, nsamples_nchannels, slice_of_data
