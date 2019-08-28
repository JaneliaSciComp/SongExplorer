#!/usr/bin/python3

# record whether annotations where correctly or mistakenly classified
 
# mistakes.py <path-to-annotations-npz-file>

# e.g.
# mistakes.py `pwd`/groundtruth-data

import os
import numpy as np
import sys
from sys import argv
from natsort import natsorted
import csv

_, groundtruth_directory = argv
print('groundtruth_directory: '+groundtruth_directory)

npzfile = np.load(os.path.join(groundtruth_directory, 'activations.npz'),
                  allow_pickle=True)
samples = npzfile['samples']
arr_ = natsorted(filter(lambda x: x.startswith('arr_'), npzfile.files))
logits = npzfile[arr_[-1]]
labels = list(npzfile['labels'])

isort = [x for x,y in sorted(enumerate(samples), key = lambda x: x[1]['ticks'][0])]

fids = []
csvwriters = {}
for idx in isort:
  wavbase,_ = os.path.splitext(samples[idx]['file'])
  if wavbase not in csvwriters:
    fids.append(open(wavbase+'-mistakes.csv', 'w', newline=''))
    csvwriters[wavbase] = csv.writer(fids[-1])
  classified_as = np.argmax(logits[idx])
  annotated_as = labels.index(samples[idx]['label'])
  csvwriters[wavbase].writerow([os.path.basename(samples[idx]['file']),
        samples[idx]['ticks'][0], samples[idx]['ticks'][1],
        'correct' if classified_as == annotated_as else 'mistaken',
        labels[classified_as],
        samples[idx]['label']])

for fid in fids:
  fid.close()
