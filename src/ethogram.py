#!/usr/bin/python3

# apply per-class thresholds to discretize probabilities
 
# ethogram.py <logdir> <model> <thresholds-file> <tf-file> <wav-tic-rate>

# e.g.
# ethogram.py `pwd`/trained-classifier 1k 50 `pwd`/groundtruth-data/round1/20161207T102314_ch1_p1.tf 5000

import sys
import os
import re
import numpy as np
import csv
from sys import argv
from scipy.io import wavfile
from itertools import cycle

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from lib import *

_,logdir,model,thresholds_file,tf_file,audio_tic_rate = argv
print('logdir: '+logdir)
print('model: '+model)
print('thresholds_file: '+thresholds_file)
print('tf_file: '+tf_file)
print('audio_tic_rate: '+audio_tic_rate)
audio_tic_rate=float(audio_tic_rate)

tfpath, tfname = os.path.split(tf_file)
tfname_noext = os.path.splitext(tfname)[0]
if os.path.isfile(os.path.join(tfpath,tfname_noext+'.wav')):
  wavname = tfname_noext+'.wav'
elif os.path.isfile(os.path.join(tfpath,tfname_noext+'.WAV')):
  wavname = tfname_noext+'.WAV'
else:
  print('cannot find corresponding WAV file')
  exit()

precision_recall_ratios, thresholds = read_thresholds(logdir, model, thresholds_file)

labels = [x[0] for x in thresholds]
thresholds = np.array([x[1:] for x in thresholds], dtype=np.float64)

audio_tic_rate_probabilities, half_stride_sec, probability_matrix = \
      read_probabilities(os.path.join(tfpath, tfname_noext), labels)

for ithreshold in range(len(precision_recall_ratios)):
  features, start_tics, stop_tics = discretize_probabilites(probability_matrix,
                                                            thresholds[:,[ithreshold]],
                                                            labels,
                                                            audio_tic_rate_probabilities,
                                                            half_stride_sec,
                                                            audio_tic_rate)
  filename = os.path.join(tfpath,
                          tfname_noext+'-predicted-'+precision_recall_ratios[ithreshold]+'pr.csv')
  isort = np.argsort(start_tics)
  with open(filename,'w') as fid:
    csvwriter = csv.writer(fid)
    csvwriter.writerows(zip(cycle([wavname]), \
                            start_tics[isort], stop_tics[isort], \
                            cycle(['predicted']),
                            features[isort]))
