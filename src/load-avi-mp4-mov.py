# video_read_plugin="load-avi-mp4-mov"
# video_read_plugin_kwargs={}

# use agg here as otherwise pims tries to open gtk
# see https://github.com/soft-matter/pims/issues/351
import matplotlib as mpl
mpl.use('Agg')
import pims

def video_read(avi_path, start_frame, stop_frame, **kw):
    data = pims.open(avi_path)

    if not start_frame: start_frame=0
    if not stop_frame: stop_frame=len(data)+1

    start_frame_clamped = max(0, start_frame)
    stop_frame_clamped = min(len(data)+1, stop_frame)

    data_sliced = data[start_frame_clamped:stop_frame_clamped]
    data_sliced.shape = [len(data_sliced), *data_sliced[0].shape]

    # data is indexed as data[iframe][iheight,iwidth,ichannel]

    return data.frame_rate, data_sliced
