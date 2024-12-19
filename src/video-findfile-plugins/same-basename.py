import os

def video_findfile(directory, wavfile):
  vids = list(filter(lambda x: x!=wavfile and
                               os.path.splitext(x)[0] == os.path.splitext(wavfile)[0] and
                               os.path.splitext(x)[1].lower() in ['.avi','.mp4','.mov'],
                     os.listdir(directory)))
  return vids[0] if len(vids)==1 else ""
