#!/usr/bin/python3

# generate .wav files containing per-class probabilities from .tf log files
 
# classify2.py <logdir> <model> <tffile> <context> <shift> <stride> [<labels>, <prevalences>]

# e.g.
# classify2.py `pwd`/trained-classifier 1k `pwd`/groundtruth-data/20161207T102314_ch1_p1.tf 204.8 0 1.6 mel-pulse,mel-sine,ambient 0.1,0.1,0.8

import sys
import os
import re
import numpy as np
from sys import argv
import ast
from scipy.io import wavfile

_,logdir,model,tffile,context_ms,shift_ms,stride_ms = argv[:7]
print('logdir: '+logdir)
print('model: '+model)
print('tffile: '+tffile)
print('context_ms: '+context_ms)
print('shift_ms: '+shift_ms)
print('stride_ms: '+stride_ms)
context_ms = float(context_ms)
shift_ms = float(shift_ms)
stride_ms = float(stride_ms)

with open(os.path.join(logdir,model,'vgg_labels.txt'), 'r') as fid:
  model_labels = fid.read().splitlines()

if len(argv)==9:
  labels,prevalences = argv[7:9]
  labels = np.array(labels.split(','))
  prevalences = np.array([float(x) for x in prevalences.split(',')])
  assert len(labels)==len(prevalences)==len(model_labels)
  assert set(labels)==set(model_labels)
  prevalences /= np.sum(prevalences)
  iimodel_labels = np.argsort(np.argsort(model_labels))
  ilabels = np.argsort(labels)
  labels = labels[ilabels][iimodel_labels]
  prevalences = prevalences[ilabels][iimodel_labels]
  assert np.all(labels==model_labels)
else:
  labels = model_labels
  prevalences = [1/len(labels) for _ in range(len(labels))]
print('labels: '+str(labels))
print('prevalences: '+str(prevalences))

npadding = round((context_ms/2+shift_ms)/stride_ms)
probability_matrix = np.zeros((npadding, len(labels)))
regex=re.compile('INFO:tensorflow:[0-9]+ms output_layer:0 (.+)')

with open(tffile) as fid:
  for line in fid:
    m = regex.match(line)
    if m==None:
      print(line)
      continue
    probabilities = ast.literal_eval(m.group(1).replace('nan','0'))
    probability_matrix = np.concatenate((probability_matrix,np.array(probabilities)))

tic_rate = round(1000/stride_ms)
if tic_rate != 1000/stride_ms:
  print('WARNING: .wav files do not support fractional sampling rates!')

denominator = np.sum(probability_matrix * prevalences, axis=1)
for ch in range(len(labels)):
  adjusted_probability = probability_matrix[:,ch] * prevalences[ch]
  adjusted_probability[npadding:] /= denominator[npadding:]
  #inotnan = ~isnan(probability_matrix(:,ch));
  waveform = adjusted_probability*np.iinfo(np.int16).max  # colon was inotnan
  filename = tffile[:-3]+'-'+labels[ch]+'.wav'
  wavfile.write(filename, tic_rate, waveform.astype('int16'))
