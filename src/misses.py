#!/usr/bin/python3

# find events that are detected but not predicted, a.k.a. false negatives
 
# misses.py <detected-and-predicted-csv-files>

# e.g.
# misses.py `pwd`/groundtruth-data/round2/20161207T102314_ch1_p1-detected.csv `pwd`/groundtruth-data/round2/20161207T102314_ch1_p1-predicted-2.0pr.csv

import os
from sys import argv
import numpy as np
import csv
from itertools import cycle

csv_files = argv[1:]
print('csv_files: '+str(csv_files))

detected_events=[]
predicted_events=[]
for csv_file in csv_files:
  with open(csv_file) as fid:
    csvreader = csv.reader(fid)
    for row in csvreader:
      if row[3]=='detected':
        detected_events.append(row)
      elif row[3]=='predicted':
        predicted_events.append(row)
      else:
        assert False

detected_wavs = list(set([x[0] for x in detected_events]))
predicted_wavs = list(set([x[0] for x in predicted_events]))
assert len(detected_wavs)==1
assert len(predicted_wavs)==1
assert detected_wavs == predicted_wavs

detected_max_time = np.max([int(x[2]) for x in detected_events])
predicted_max_time = np.max([int(x[2]) for x in predicted_events])
max_time = max(detected_max_time, predicted_max_time)

detected_bool = np.full((max_time,), False)
for event in detected_events:
  detected_bool[int(event[1]):int(event[2])+1] = True

predicted_bool = np.full((max_time,), False)
for event in predicted_events:
  predicted_bool[int(event[1]):int(event[2])+1] = True

misses_bool = np.logical_and(detected_bool, np.logical_not(predicted_bool))

diff_misses_bool = np.diff(misses_bool)
changes = np.where(diff_misses_bool)[0]
nfeatures = int(np.ceil(len(changes)/2))
start_times = np.empty((nfeatures,), dtype=np.int32)
stop_times = np.empty((nfeatures,), dtype=np.int32)
ifeature = 0
ichange = 1
while ichange<len(changes):
  if not misses_bool[changes[ichange]]:  # starts with word
     ichange += 1;
     continue
  start_times[ifeature] = changes[ichange-1]+1
  stop_times[ifeature] = changes[ichange]
  ifeature += 1
  ichange += 2

noext = os.path.splitext(detected_wavs[0])[0]
basepath = os.path.dirname(csv_files[0])
filename = os.path.join(basepath, noext+'-missed.csv')
with open(filename,'w') as fid:
  csvwriter = csv.writer(fid)
  csvwriter.writerows(zip(cycle([detected_wavs[0]]), \
                          start_times[:ifeature], stop_times[:ifeature], \
                          cycle(['missed']), cycle(['other'])))
