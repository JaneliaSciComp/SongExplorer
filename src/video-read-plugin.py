# a function that inputs the full path to a file containing the video
# recording, an interval of time, and some keyword arguments and returns
# the frame rate and data in time x height x width x channel format
def video_read(fullpath, start_frame, stop_frame, **kw):

    # load data, determine frame rate, and do any special processing

    # see load-avi-mp4-mov.py for a working example

    return framerate, shape, dtype, data
