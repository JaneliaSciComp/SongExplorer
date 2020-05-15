#!/usr/bin/python3

# apply per-class thresholds to discretize probabilities
 
# ethogram.py <logdir> <model> <check-point> <tffile> <wav-tic-rate>

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

_,logdir,model,check_point,tffile,audio_tic_rate = argv
print('logdir: '+logdir)
print('model: '+model)
print('check_point: '+check_point)
print('tffile: '+tffile)
print('audio_tic_rate: '+audio_tic_rate)
audio_tic_rate=float(audio_tic_rate)

with open(os.path.join(logdir,model,'vgg_labels.txt'), 'r') as fid:
  labels = fid.read().splitlines()

tfpath, tfname = os.path.split(tffile)
tfname_noext = os.path.splitext(tfname)[0]
if os.path.isfile(os.path.join(tfpath,tfname_noext+'.wav')):
  wavname = tfname_noext+'.wav'
elif os.path.isfile(os.path.join(tfpath,tfname_noext+'.WAV')):
  wavname = tfname_noext+'.WAV'
else:
  print('cannot find corresponding WAV file')
  exit()

precision_recall_ratios=None
thresholds=[]
with open(os.path.join(logdir,model,'thresholds.ckpt-'+check_point+'.csv')) as fid:
  csvreader = csv.reader(fid)
  for row in csvreader:
    if precision_recall_ratios==None:
      precision_recall_ratios=row[1:]
    else:
      thresholds.append(row)

assert [x[0] for x in thresholds]==labels
thresholds = np.array([x[1:] for x in thresholds], dtype=np.float64)

wavpath = os.path.join(tfpath, tfname_noext+'-'+labels[0]+'.wav')
sample_rate_probabilities, probabilities = wavfile.read(wavpath)
half_stride_sec = 1/sample_rate_probabilities/2
probability_matrix = np.empty((len(labels), len(probabilities)))
probability_matrix[0,:] = probabilities / np.iinfo(probabilities.dtype).max
for ilabel in range(1,len(labels)):
  wavpath = os.path.join(tfpath, tfname_noext+'-'+labels[ilabel]+'.wav')
  sample_rate_probabilities, probabilities = wavfile.read(wavpath)
  probability_matrix[ilabel,:] = probabilities / np.iinfo(probabilities.dtype).max

for ithreshold in range(len(precision_recall_ratios)):
  behavior = probability_matrix > thresholds[:,[ithreshold]]
  diff_behavior = np.diff(behavior)
  ichanges, jchanges = np.where(diff_behavior)
  nfeatures = int(np.ceil(len(ichanges)/2))
  features = np.empty((nfeatures,), dtype=object)
  start_tics = np.empty((nfeatures,), dtype=np.int32)
  stop_tics = np.empty((nfeatures,), dtype=np.int32)
  ifeature = 0
  ijchange = 1
  while ijchange<len(ichanges):                 # spans classes or starts with word
    if ichanges[ijchange-1]!=ichanges[ijchange] or not behavior[ichanges[ijchange],jchanges[ijchange]]:
       ijchange += 1;
       continue
    start_tics[ifeature] = jchanges[ijchange-1] + 1
    stop_tics[ifeature] = jchanges[ijchange]
    features[ifeature] = labels[ichanges[ijchange]]
    ifeature += 1
    ijchange += 2
  ifeature -= 1
  #
  features = features[:ifeature]
  start_tics = np.round((start_tics[:ifeature] / sample_rate_probabilities \
                         - half_stride_sec) \
                        * audio_tic_rate).astype(np.int)
  stop_tics = np.round((stop_tics[:ifeature] / sample_rate_probabilities \
                         + half_stride_sec) \
                       * audio_tic_rate).astype(np.int)
  #
  isort = np.argsort(start_tics)
  filename=os.path.join(tfpath, tfname_noext+'-predicted-'+precision_recall_ratios[ithreshold]+'pr.csv')
  with open(filename,'w') as fid:
    csvwriter = csv.writer(fid)
    csvwriter.writerows(zip(cycle([wavname]), \
                            start_tics[isort], \
                            stop_tics[isort], \
                            cycle(['predicted']),features[isort]))
