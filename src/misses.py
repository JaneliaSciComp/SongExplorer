#!/usr/bin/python3

# find events that are detected but not predicted, a.k.a. false negatives
 
# misses.py <detected-and-predicted-csv-files>

# e.g.
# misses.py `pwd`/groundtruth-data/round2/20161207T102314_ch1_p1-detected.csv `pwd`/groundtruth-data/round2/20161207T102314_ch1_p1-predicted-2.0pr.csv

import os
import sys
import numpy as np
import csv
from itertools import cycle

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from lib import *

csv_files = sys.argv[1:]
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

start_times, stop_times, ifeature = combine_events(detected_events, predicted_events,
      lambda x,y: np.logical_and(x, np.logical_not(y)))

noext = os.path.splitext(detected_wavs[0])[0]
basepath = os.path.dirname(csv_files[0])
filename = os.path.join(basepath, noext+'-missed.csv')
with open(filename,'w') as fid:
  csvwriter = csv.writer(fid)
  csvwriter.writerows(zip(cycle([detected_wavs[0]]), \
                          start_times[:ifeature], stop_times[:ifeature], \
                          cycle(['missed']), cycle(['other'])))
