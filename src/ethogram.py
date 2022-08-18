#!/usr/bin/env python3

# apply per-class thresholds to discretize probabilities
 
# ethogram.py <logdir> <model> <thresholds-file> <wav-file> <wav-tic-rate>

# e.g.
# ethogram.py `pwd`/trained-classifier 1k 50 `pwd`/groundtruth-data/round1/20161207T102314_ch1_p1.wav 5000

import sys
import os
import re
import numpy as np
import csv
from sys import argv
from scipy.io import wavfile
from itertools import cycle
from datetime import datetime
import socket

repodir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

sys.path.append(os.path.join(repodir, "src"))
from lib import *

print(str(datetime.now())+": start time")
with open(os.path.join(repodir, "VERSION.txt"), 'r') as fid:
  print('SongExplorer version = '+fid.read().strip().replace('\n',', '))
print("hostname = "+socket.gethostname())

try:

  _,logdir,model,thresholds_file,wav_file,audio_tic_rate = argv
  print('logdir: '+logdir)
  print('model: '+model)
  print('thresholds_file: '+thresholds_file)
  print('wav_file: '+wav_file)
  print('audio_tic_rate: '+audio_tic_rate)
  audio_tic_rate=float(audio_tic_rate)

  if not os.path.isfile(wav_file):
    print('cannot find WAV file')
    exit()
  wavpath, wavname = os.path.split(wav_file)
  wavname_noext = os.path.splitext(wavname)[0]

  precision_recall_ratios, thresholds = read_thresholds(logdir, model, thresholds_file)

  labels = [x[0] for x in thresholds]
  thresholds = np.array([x[1:] for x in thresholds], dtype=np.float64)

  audio_tic_rate_probabilities, half_stride_sec, probability_matrix = \
        read_probabilities(os.path.join(wavpath, wavname_noext), labels)

  for ithreshold in range(len(precision_recall_ratios)):
    features, start_tics, stop_tics = discretize_probabilities(probability_matrix,
                                                               thresholds[:,[ithreshold]],
                                                               labels,
                                                               audio_tic_rate_probabilities,
                                                               half_stride_sec,
                                                               audio_tic_rate)
    filename = os.path.join(wavpath,
                            wavname_noext+'-predicted-'+precision_recall_ratios[ithreshold]+'pr.csv')
    isort = np.argsort(start_tics)
    with open(filename,'w') as fid:
      csvwriter = csv.writer(fid)
      csvwriter.writerows(zip(cycle([wavname]), \
                              start_tics[isort], stop_tics[isort], \
                              cycle(['predicted']),
                              features[isort]))

except Exception as e:
  print(e)

finally:
  os.sync()
  print(str(datetime.now())+": finish time")
