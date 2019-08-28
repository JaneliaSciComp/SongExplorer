#!/usr/bin/python3

# generate confusion matrices, precision-recall curves, thresholds, etc.
 
# probabilities.py <logdir> <model> <tffile> <context> <shift> <stride>

# e.g.
# deepsong probabilities.py `pwd`/trained-classifier 1k `pwd`/groundtruth-data/20161207T102314_ch1_p1.tf 204.8 0 1.6

import sys
import os
import re
import numpy as np
from sys import argv
import ast
from scipy.io import wavfile

_,logdir,model,tffile,context_ms,shift_ms,stride_ms = argv
print('logdir: '+logdir)
print('model: '+model)
print('tffile: '+tffile)
print('context_ms: '+context_ms)
print('shift_ms: '+shift_ms)
print('stride_ms: '+stride_ms)
context_ms = float(context_ms)
shift_ms = float(shift_ms)
stride_ms = float(stride_ms)

with open(os.path.join(logdir,'train_'+model,'vgg_labels.txt'), 'r') as fid:
  labels = fid.read().splitlines()

padding = round((context_ms/2+shift_ms)/stride_ms)
probability_matrix = np.zeros((padding, len(labels)))
regex=re.compile('.+] [0-9]+ms: output_layer (.+)')

with open(tffile) as fid:
  for line in fid:
    m = regex.match(line)
    if m==None:
      print(line)
      continue
    probabilities = ast.literal_eval('['+m.group(1).replace(' ',', ').replace('][','],[').replace('nan','0')+']')
    probability_matrix = np.concatenate((probability_matrix,np.array(probabilities)))

Fs = round(1000/stride_ms)
if Fs != 1000/stride_ms:
  print('WARNING: .wav files do not support fractional sampling rates!')

for ch in range(len(labels)):
  #inotnan = ~isnan(probability_matrix(:,ch));
  filename = tffile[:-3]+'-'+labels[ch]+'.wav'
  waveform = probability_matrix[:,ch]*np.iinfo(np.int16).max  # colon was inotnan
  wavfile.write(filename, Fs, waveform.astype('int16'))
