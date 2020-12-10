#!/usr/bin/python3

# test doing, undoing, and redoing annotations

# export SINGULARITYENV_SONGEXPLORER_STATE=/tmp
# ${SONGEXPLORER_BIN/-B/-B /tmp:/opt/songexplorer/test/scratch -B} test/annotating.py

import sys
import os
import shutil
import glob
from subprocess import run, PIPE, STDOUT, Popen
import time
import math
import csv
import numpy as np

from lib import count_lines, check_file_exists

def check_annotation_in_memory(thissample, shouldexist):
  if shouldexist:
    if len(M.isannotated(thissample))==0:
      print("ERROR: annotation is not in memory")
    if thissample not in M.annotated_samples:
      print("ERROR: annotation is not in memory")
  else:
    if len(M.isannotated(thissample))>0:
      print("ERROR: annotation is in memory")
    if thissample in M.annotated_samples:
      print("ERROR: annotation is in memory")

def check_annotation_on_disk(csvfile, thissample, shouldexist):
  with open(csvfile) as fid:
    csvreader = csv.reader(fid)
    foundit=False
    for line in csvreader:
      if line[0]==os.path.basename(thissample['file']) and \
         line[1]==str(thissample['ticks'][0]) and \
         line[2]==str(thissample['ticks'][1]) and \
         line[3]=='annotated' and \
         line[4]==thissample['label']:
         foundit=True
         break
    if shouldexist and not foundit:
      print("ERROR: annotation is not on disk")
    if not shouldexist and foundit:
      print("ERROR: annotation is on disk")

repo_path = os.path.dirname(sys.path[0])
  
sys.path.append(os.path.join(repo_path, "src/gui"))
import model as M
import view as V
import controller as C

os.makedirs(os.path.join(repo_path, "test/scratch/annotating"))
shutil.copy(os.path.join(repo_path, "configuration.pysh"),
            os.path.join(repo_path, "test/scratch/annotating"))

M.init(None, os.path.join(repo_path, "test/scratch/annotating/configuration.pysh"))
V.init(None)
C.init(None)

basepath = os.path.join(repo_path, "test/scratch/annotating/groundtruth-data/round1")
os.makedirs(basepath)
shutil.copy(os.path.join(repo_path, "data/PS_20130625111709_ch3.wav"), basepath)

wavfile = os.path.join(basepath, 'PS_20130625111709_ch3.wav')
M.clustered_samples = [
  {'label': 'time', 'file': wavfile, 'ticks': [2, 5], 'kind': 'detected'},
  {'label': 'frequency', 'file': wavfile, 'ticks': [10, 15], 'kind': 'detected'},
  {'label': 'neither', 'file': wavfile, 'ticks': [20, 30], 'kind': 'detected'},
  ]
M.clustered_starts_sorted = [x['ticks'][0] for x in M.clustered_samples]
M.clustered_stops = [x['ticks'][1] for x in M.clustered_samples]
M.iclustered_stops_sorted = np.argsort(M.clustered_stops)
M.state['labels'] = ['pulse', 'sine', 'ambient', '']

# add pulse
thissample = {'file': wavfile, 'ticks': [3, 3], 'label': 'pulse'}
M.add_annotation(thissample)
check_annotation_in_memory(thissample, True)
M.save_annotations()
files = os.listdir(basepath)
if len(files)!=2: print("ERROR: wrong number of files")

csvfile = os.path.join(basepath, next(filter(lambda x: x.endswith('.csv'), files)))
count_lines(csvfile, 1)
check_annotation_on_disk(csvfile, thissample, True)

# undo it
C.undo_callback()
check_annotation_in_memory(thissample, False)
M.save_annotations()
files = os.listdir(basepath)
if len(files)!=1: print("ERROR: wrong number of files")

# redo it
C.redo_callback()
check_annotation_in_memory(thissample, True)
M.save_annotations()
files = os.listdir(basepath)
if len(files)!=2: print("ERROR: wrong number of files")

csvfile = os.path.join(basepath, next(filter(lambda x: x.endswith('.csv'), files)))
count_lines(csvfile, 1)
check_annotation_on_disk(csvfile, thissample, True)

# add sine
thissample = {'file': wavfile, 'ticks': [12, 16], 'label': 'sine'}
M.add_annotation(thissample)
check_annotation_in_memory(thissample, True)
M.save_annotations()
files = os.listdir(basepath)
if len(files)!=2: print("ERROR: wrong number of files")

csvfile = os.path.join(basepath, next(filter(lambda x: x.endswith('.csv'), files)))
count_lines(csvfile, 2)
check_annotation_on_disk(csvfile, thissample, True)

# add ambient
thissample = {'file': wavfile, 'ticks': [19, 29], 'label': 'ambient'}
M.add_annotation(thissample)
check_annotation_in_memory(thissample, True)
M.save_annotations()
files = os.listdir(basepath)
if len(files)!=2: print("ERROR: wrong number of files")

csvfile = os.path.join(basepath, next(filter(lambda x: x.endswith('.csv'), files)))
count_lines(csvfile, 3)
check_annotation_on_disk(csvfile, thissample, True)

# undo it
C.undo_callback()
check_annotation_in_memory(thissample, False)
M.save_annotations()
files = os.listdir(basepath)
if len(files)!=2: print("ERROR: wrong number of files")

csvfile = os.path.join(basepath, next(filter(lambda x: x.endswith('.csv'), files)))
count_lines(csvfile, 2)
check_annotation_on_disk(csvfile, thissample, False)

# redo it
C.redo_callback()
check_annotation_in_memory(thissample, True)
M.save_annotations()
files = os.listdir(basepath)
if len(files)!=2: print("ERROR: wrong number of files")

csvfile = os.path.join(basepath, next(filter(lambda x: x.endswith('.csv'), files)))
count_lines(csvfile, 3)
check_annotation_on_disk(csvfile, thissample, True)

# undo all
C.undo_callback()
C.undo_callback()
C.undo_callback()
if len(M.annotated_samples)!=0: print("ERROR: wrong number of elements")

M.save_annotations()
files = os.listdir(basepath)
if len(files)!=1: print("ERROR: wrong number of files")

# toggle in time
M.ilabel=0
C.toggle_annotation(0)
thissample = M.clustered_samples[0].copy()
thissample.pop('kind', None)
thissample['label']=M.state['labels'][M.ilabel]
check_annotation_in_memory(thissample, True)
M.save_annotations()
files = os.listdir(basepath)
if len(files)!=2: print("ERROR: wrong number of files")

csvfile = os.path.join(basepath, next(filter(lambda x: x.endswith('.csv'), files)))
count_lines(csvfile, 1)
check_annotation_on_disk(csvfile, thissample, True)

# undo it
C.undo_callback()
check_annotation_in_memory(thissample, False)
M.save_annotations()
files = os.listdir(basepath)
if len(files)!=1: print("ERROR: wrong number of files")

# redo it
C.redo_callback()
check_annotation_in_memory(thissample, True)
M.save_annotations()
files = os.listdir(basepath)
if len(files)!=2: print("ERROR: wrong number of files")

csvfile = os.path.join(basepath, next(filter(lambda x: x.endswith('.csv'), files)))
count_lines(csvfile, 1)
check_annotation_on_disk(csvfile, thissample, True)

# toggle out time
C.toggle_annotation(0)
check_annotation_in_memory(thissample, False)
M.save_annotations()
files = os.listdir(basepath)
if len(files)!=1: print("ERROR: wrong number of files")

# toggle in time
C.toggle_annotation(0)
check_annotation_in_memory(thissample, True)
M.save_annotations()
files = os.listdir(basepath)
if len(files)!=2: print("ERROR: wrong number of files")

csvfile = os.path.join(basepath, next(filter(lambda x: x.endswith('.csv'), files)))
count_lines(csvfile, 1)
check_annotation_on_disk(csvfile, thissample, True)

# toggle in frequency
M.ilabel=1
C.toggle_annotation(1)
thissample = M.clustered_samples[1].copy()
thissample.pop('kind', None)
thissample['label']=M.state['labels'][M.ilabel]
check_annotation_in_memory(thissample, True)
M.save_annotations()
files = os.listdir(basepath)
if len(files)!=2: print("ERROR: wrong number of files")

csvfile = os.path.join(basepath, next(filter(lambda x: x.endswith('.csv'), files)))
count_lines(csvfile, 2)
check_annotation_on_disk(csvfile, thissample, True)

# toggle in ambient
M.ilabel=2
C.toggle_annotation(2)
thissample = M.clustered_samples[2].copy()
thissample.pop('kind', None)
thissample['label']=M.state['labels'][M.ilabel]
check_annotation_in_memory(thissample, True)
M.save_annotations()
files = os.listdir(basepath)
if len(files)!=2: print("ERROR: wrong number of files")

csvfile = os.path.join(basepath, next(filter(lambda x: x.endswith('.csv'), files)))
count_lines(csvfile, 3)
check_annotation_on_disk(csvfile, thissample, True)

# undo it
C.undo_callback()
check_annotation_in_memory(thissample, False)
M.save_annotations()
files = os.listdir(basepath)
if len(files)!=2: print("ERROR: wrong number of files")

csvfile = os.path.join(basepath, next(filter(lambda x: x.endswith('.csv'), files)))
count_lines(csvfile, 2)
check_annotation_on_disk(csvfile, thissample, False)

# redo it
C.redo_callback()
check_annotation_in_memory(thissample, True)
M.save_annotations()
files = os.listdir(basepath)
if len(files)!=2: print("ERROR: wrong number of files")

csvfile = os.path.join(basepath, next(filter(lambda x: x.endswith('.csv'), files)))
count_lines(csvfile, 3)
check_annotation_on_disk(csvfile, thissample, True)

# toggle out ambient
C.toggle_annotation(2)
check_annotation_in_memory(thissample, False)
M.save_annotations()
files = os.listdir(basepath)
if len(files)!=2: print("ERROR: wrong number of files")

csvfile = os.path.join(basepath, next(filter(lambda x: x.endswith('.csv'), files)))
count_lines(csvfile, 2)
check_annotation_on_disk(csvfile, thissample, False)

# toggle in ambient
M.ilabel=2
C.toggle_annotation(2)
thissample = M.clustered_samples[2].copy()
thissample.pop('kind', None)
thissample['label']=M.state['labels'][M.ilabel]
check_annotation_in_memory(thissample, True)
M.save_annotations()
files = os.listdir(basepath)
if len(files)!=2: print("ERROR: wrong number of files")

csvfile = os.path.join(basepath, next(filter(lambda x: x.endswith('.csv'), files)))
count_lines(csvfile, 3)
check_annotation_on_disk(csvfile, thissample, True)

# undo all
C.undo_callback()
C.undo_callback()
C.undo_callback()
C.undo_callback()
C.undo_callback()
if len(M.annotated_samples)!=0: print("ERROR: wrong number of elements")

M.save_annotations()
files = os.listdir(basepath)
if len(files)!=1: print("ERROR: wrong number of files")

##
M.clustered_samples = [
  {'label': 'pulse', 'file': wavfile, 'ticks': [2, 5], 'kind': 'annotated'},
  {'label': 'sine', 'file': wavfile, 'ticks': [10, 15], 'kind': 'annotated'},
  {'label': 'ambient', 'file': wavfile, 'ticks': [20, 30], 'kind': 'annotated'},
  ]
M.clustered_starts_sorted = [x['ticks'][0] for x in M.clustered_samples]
M.clustered_stops = [x['ticks'][1] for x in M.clustered_samples]
M.iclustered_stops_sorted = np.argsort(M.clustered_stops)
csvfile = os.path.join(basepath, 'PS_20130625111709_ch3-annotated-original.csv')
with open(csvfile, 'w') as fid:
  csvwriter = csv.writer(fid)
  for sample in M.clustered_samples:
    csvwriter.writerow([os.path.basename(sample['file']),
                        sample['ticks'][0], sample['ticks'][1],
                        sample['kind'],
                        sample['label']])

# pulse to ambient
M.ilabel=2
C.toggle_annotation(0)
thissample = M.clustered_samples[0].copy()
thissample.pop('kind', None)
thissample['label']=M.state['labels'][M.ilabel]
check_annotation_in_memory(thissample, True)
M.save_annotations()
files = os.listdir(basepath)
if len(files)!=3: print("ERROR: wrong number of files")

thissample = M.clustered_samples[0].copy()
thissample.pop('kind', None)
csvfile = os.path.join(basepath,
                       next(filter(lambda x: x.endswith('.csv') and 'original' in x,
                                   files)))
count_lines(csvfile, 2)
check_annotation_on_disk(csvfile, thissample, False)

thissample['label']=M.state['labels'][M.ilabel]
csvfile = os.path.join(basepath,
                       next(filter(lambda x: x.endswith('.csv') and 'original' not in x,
                                   files)))
count_lines(csvfile, 1)
check_annotation_on_disk(csvfile, thissample, True)

#undo
C.undo_callback()
thissample = M.clustered_samples[0].copy()
thissample.pop('kind', None)
check_annotation_in_memory(thissample, True)
M.save_annotations()
files = os.listdir(basepath)
if len(files)!=3: print("ERROR: wrong number of files")

csvfile = os.path.join(basepath,
                       next(filter(lambda x: x.endswith('.csv') and 'original' in x,
                                   files)))
count_lines(csvfile, 2)
check_annotation_on_disk(csvfile, thissample, False)

csvfile = os.path.join(basepath,
                       next(filter(lambda x: x.endswith('.csv') and 'original' not in x,
                                   files)))
count_lines(csvfile, 1)
check_annotation_on_disk(csvfile, thissample, True)

#redo
C.redo_callback()
thissample = M.clustered_samples[0].copy()
thissample.pop('kind', None)
thissample['label']=M.state['labels'][M.ilabel]
check_annotation_in_memory(thissample, True)
M.save_annotations()
files = os.listdir(basepath)
if len(files)!=3: print("ERROR: wrong number of files")

thissample = M.clustered_samples[0].copy()
thissample.pop('kind', None)
csvfile = os.path.join(basepath,
                       next(filter(lambda x: x.endswith('.csv') and 'original' in x,
                                   files)))
count_lines(csvfile, 2)
check_annotation_on_disk(csvfile, thissample, False)

thissample['label']=M.state['labels'][M.ilabel]
csvfile = os.path.join(basepath,
                       next(filter(lambda x: x.endswith('.csv') and 'original' not in x,
                                   files)))
count_lines(csvfile, 1)
check_annotation_on_disk(csvfile, thissample, True)

# delete sine
M.ilabel=3
C.toggle_annotation(1)
thissample = M.clustered_samples[1].copy()
thissample.pop('kind', None)
thissample['label']=M.state['labels'][M.ilabel]
check_annotation_in_memory(thissample, True)
M.save_annotations()
files = os.listdir(basepath)
if len(files)!=3: print("ERROR: wrong number of files")

thissample = M.clustered_samples[1].copy()
thissample.pop('kind', None)
csvfile = os.path.join(basepath,
                       next(filter(lambda x: x.endswith('.csv') and 'original' in x,
                                   files)))
count_lines(csvfile, 1)
check_annotation_on_disk(csvfile, thissample, False)

csvfile = os.path.join(basepath,
                       next(filter(lambda x: x.endswith('.csv') and 'original' not in x,
                                   files)))
count_lines(csvfile, 1)
check_annotation_on_disk(csvfile, thissample, False)

#undo
C.undo_callback()
thissample = M.clustered_samples[1].copy()
thissample.pop('kind', None)
check_annotation_in_memory(thissample, True)
M.save_annotations()
files = os.listdir(basepath)
if len(files)!=3: print("ERROR: wrong number of files")

csvfile = os.path.join(basepath,
                       next(filter(lambda x: x.endswith('.csv') and 'original' in x,
                                   files)))
count_lines(csvfile, 1)
check_annotation_on_disk(csvfile, thissample, False)

csvfile = os.path.join(basepath,
                       next(filter(lambda x: x.endswith('.csv') and 'original' not in x,
                                   files)))
count_lines(csvfile, 2)
check_annotation_on_disk(csvfile, thissample, True)

#redo
C.redo_callback()
thissample = M.clustered_samples[1].copy()
thissample.pop('kind', None)
thissample['label']=M.state['labels'][M.ilabel]
check_annotation_in_memory(thissample, True)
M.save_annotations()
files = os.listdir(basepath)
if len(files)!=3: print("ERROR: wrong number of files")

thissample = M.clustered_samples[1].copy()
thissample.pop('kind', None)
csvfile = os.path.join(basepath,
                       next(filter(lambda x: x.endswith('.csv') and 'original' in x,
                                   files)))
count_lines(csvfile, 1)
check_annotation_on_disk(csvfile, thissample, False)

csvfile = os.path.join(basepath,
                       next(filter(lambda x: x.endswith('.csv') and 'original' not in x,
                                   files)))
count_lines(csvfile, 1)
check_annotation_on_disk(csvfile, thissample, False)

# ambient to pulse
M.ilabel=0
C.toggle_annotation(2)
thissample = M.clustered_samples[2].copy()
thissample.pop('kind', None)
thissample['label']=M.state['labels'][M.ilabel]
check_annotation_in_memory(thissample, True)
M.save_annotations()
files = os.listdir(basepath)
if len(files)!=2: print("ERROR: wrong number of files")

thissample['label']=M.state['labels'][M.ilabel]
csvfile = os.path.join(basepath,
                       next(filter(lambda x: x.endswith('.csv') and 'original' not in x,
                                   files)))
count_lines(csvfile, 2)
check_annotation_on_disk(csvfile, thissample, True)

#undo
C.undo_callback()
thissample = M.clustered_samples[2].copy()
thissample.pop('kind', None)
check_annotation_in_memory(thissample, True)
M.save_annotations()
files = os.listdir(basepath)
if len(files)!=2: print("ERROR: wrong number of files")

csvfile = os.path.join(basepath,
                       next(filter(lambda x: x.endswith('.csv') and 'original' not in x,
                                   files)))
count_lines(csvfile, 2)
check_annotation_on_disk(csvfile, thissample, True)

#redo
C.redo_callback()
thissample = M.clustered_samples[2].copy()
thissample.pop('kind', None)
thissample['label']=M.state['labels'][M.ilabel]
check_annotation_in_memory(thissample, True)
M.save_annotations()
files = os.listdir(basepath)
if len(files)!=2: print("ERROR: wrong number of files")

csvfile = os.path.join(basepath,
                       next(filter(lambda x: x.endswith('.csv') and 'original' not in x,
                                   files)))
count_lines(csvfile, 2)
check_annotation_on_disk(csvfile, thissample, True)
