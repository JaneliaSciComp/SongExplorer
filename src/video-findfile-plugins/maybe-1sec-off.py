import os
import datetime as dt
import logging

def within_X_seconds(wavdate, vidfile, X):
  viddate = dt.datetime.strptime(vidfile[:15], "%Y%m%d_%H%M%S")
  return abs(viddate-wavdate) <= dt.timedelta(seconds=X)

def video_findfile(directory, wavfile):
  try:
      wavdate = dt.datetime.strptime(wavfile[:15], "%Y%m%d_%H%M%S")
  except ValueError:
      print("ERROR: "+wavfile+" does not have a timestamp in the filename of the form %Y%m%d_%H%M%S")
      return ""
  vids = list(filter(lambda x: x!=wavfile and
                               x[15:-4] == wavfile[15:-4] and
                               within_X_seconds(wavdate, x, 1) and
                               os.path.splitext(x)[1].lower() in ['.avi','.mp4','.mov'],
                     os.listdir(directory)))
  return vids[0] if len(vids)==1 else ""
