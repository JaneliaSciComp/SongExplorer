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

detected_events = {}
predicted_events = {}
for csv_file in csv_files:
  with open(csv_file) as fid:
    csvreader = csv.reader(fid)
    for row in csvreader:
      if row[3]=='detected':
        if row[0] not in detected_events:
          detected_events[row[0]] = []
        detected_events[row[0]].append(row)
      elif row[3]=='predicted':
        if row[0] not in predicted_events:
          predicted_events[row[0]] = []
        predicted_events[row[0]].append(row)
      else:
        assert False

assert detected_events.keys() == predicted_events.keys()

for wavfile in detected_events.keys():
  start_times, stop_times, ifeature = combine_events(
        detected_events[wavfile], predicted_events[wavfile],
        lambda x,y: np.logical_and(x, np.logical_not(y)))

  noext = os.path.splitext(wavfile)[0]
  basepath = os.path.dirname(csv_files[0])
  filename = os.path.join(basepath, noext+'-missed.csv')
  with open(filename,'w') as fid:
    csvwriter = csv.writer(fid)
    csvwriter.writerows(zip(cycle([wavfile]), \
                            start_times[:ifeature], stop_times[:ifeature], \
                            cycle(['missed']), cycle(['other'])))
