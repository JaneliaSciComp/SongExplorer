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

def check_annotation_in_memory(thissound, shouldexist):
  if shouldexist:
    if len(M.isannotated(thissound))==0:
      print("ERROR: annotation is not in memory")
    if thissound not in M.annotated_sounds:
      print("ERROR: annotation is not in memory")
  else:
    if len(M.isannotated(thissound))>0:
      print("ERROR: annotation is in memory")
    if thissound in M.annotated_sounds:
      print("ERROR: annotation is in memory")

def check_annotation_on_disk(csvfile, thissound, shouldexist):
  with open(csvfile) as fid:
    csvreader = csv.reader(fid)
    foundit=False
    for line in csvreader:
      if line[0]==os.path.basename(thissound['file']) and \
         line[1]==str(thissound['ticks'][0]) and \
         line[2]==str(thissound['ticks'][1]) and \
         line[3]=='annotated' and \
         line[4]==thissound['label']:
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
M.clustered_sounds = [
  {'label': 'time', 'file': wavfile, 'ticks': [2, 5], 'kind': 'detected'},
  {'label': 'frequency', 'file': wavfile, 'ticks': [10, 15], 'kind': 'detected'},
  {'label': 'neither', 'file': wavfile, 'ticks': [20, 30], 'kind': 'detected'},
  ]
M.clustered_starts_sorted = [x['ticks'][0] for x in M.clustered_sounds]
M.clustered_stops = [x['ticks'][1] for x in M.clustered_sounds]
M.iclustered_stops_sorted = np.argsort(M.clustered_stops)
M.state['labels'] = ['pulse', 'sine', 'ambient', '']

# add pulse
thissound = {'file': wavfile, 'ticks': [3, 3], 'label': 'pulse'}
M.add_annotation(thissound)
check_annotation_in_memory(thissound, True)
M.save_annotations()
files = os.listdir(basepath)
if len(files)!=2: print("ERROR: wrong number of files")

csvfile = os.path.join(basepath, next(filter(lambda x: x.endswith('.csv'), files)))
count_lines(csvfile, 1)
check_annotation_on_disk(csvfile, thissound, True)

# undo it
C.undo_callback()
check_annotation_in_memory(thissound, False)
M.save_annotations()
files = os.listdir(basepath)
if len(files)!=1: print("ERROR: wrong number of files")

# redo it
C.redo_callback()
check_annotation_in_memory(thissound, True)
M.save_annotations()
files = os.listdir(basepath)
if len(files)!=2: print("ERROR: wrong number of files")

csvfile = os.path.join(basepath, next(filter(lambda x: x.endswith('.csv'), files)))
count_lines(csvfile, 1)
check_annotation_on_disk(csvfile, thissound, True)

# add sine
thissound = {'file': wavfile, 'ticks': [12, 16], 'label': 'sine'}
M.add_annotation(thissound)
check_annotation_in_memory(thissound, True)
M.save_annotations()
files = os.listdir(basepath)
if len(files)!=2: print("ERROR: wrong number of files")

csvfile = os.path.join(basepath, next(filter(lambda x: x.endswith('.csv'), files)))
count_lines(csvfile, 2)
check_annotation_on_disk(csvfile, thissound, True)

# add ambient
thissound = {'file': wavfile, 'ticks': [19, 29], 'label': 'ambient'}
M.add_annotation(thissound)
check_annotation_in_memory(thissound, True)
M.save_annotations()
files = os.listdir(basepath)
if len(files)!=2: print("ERROR: wrong number of files")

csvfile = os.path.join(basepath, next(filter(lambda x: x.endswith('.csv'), files)))
count_lines(csvfile, 3)
check_annotation_on_disk(csvfile, thissound, True)

# undo it
C.undo_callback()
check_annotation_in_memory(thissound, False)
M.save_annotations()
files = os.listdir(basepath)
if len(files)!=2: print("ERROR: wrong number of files")

csvfile = os.path.join(basepath, next(filter(lambda x: x.endswith('.csv'), files)))
count_lines(csvfile, 2)
check_annotation_on_disk(csvfile, thissound, False)

# redo it
C.redo_callback()
check_annotation_in_memory(thissound, True)
M.save_annotations()
files = os.listdir(basepath)
if len(files)!=2: print("ERROR: wrong number of files")

csvfile = os.path.join(basepath, next(filter(lambda x: x.endswith('.csv'), files)))
count_lines(csvfile, 3)
check_annotation_on_disk(csvfile, thissound, True)

# undo all
C.undo_callback()
C.undo_callback()
C.undo_callback()
if len(M.annotated_sounds)!=0: print("ERROR: wrong number of elements")

M.save_annotations()
files = os.listdir(basepath)
if len(files)!=1: print("ERROR: wrong number of files")

# toggle in time
M.ilabel=0
C.toggle_annotation(0)
thissound = M.clustered_sounds[0].copy()
thissound.pop('kind', None)
thissound['label']=M.state['labels'][M.ilabel]
check_annotation_in_memory(thissound, True)
M.save_annotations()
files = os.listdir(basepath)
if len(files)!=2: print("ERROR: wrong number of files")

csvfile = os.path.join(basepath, next(filter(lambda x: x.endswith('.csv'), files)))
count_lines(csvfile, 1)
check_annotation_on_disk(csvfile, thissound, True)

# undo it
C.undo_callback()
check_annotation_in_memory(thissound, False)
M.save_annotations()
files = os.listdir(basepath)
if len(files)!=1: print("ERROR: wrong number of files")

# redo it
C.redo_callback()
check_annotation_in_memory(thissound, True)
M.save_annotations()
files = os.listdir(basepath)
if len(files)!=2: print("ERROR: wrong number of files")

csvfile = os.path.join(basepath, next(filter(lambda x: x.endswith('.csv'), files)))
count_lines(csvfile, 1)
check_annotation_on_disk(csvfile, thissound, True)

# toggle out time
C.toggle_annotation(0)
check_annotation_in_memory(thissound, False)
M.save_annotations()
files = os.listdir(basepath)
if len(files)!=1: print("ERROR: wrong number of files")

# toggle in time
C.toggle_annotation(0)
check_annotation_in_memory(thissound, True)
M.save_annotations()
files = os.listdir(basepath)
if len(files)!=2: print("ERROR: wrong number of files")

csvfile = os.path.join(basepath, next(filter(lambda x: x.endswith('.csv'), files)))
count_lines(csvfile, 1)
check_annotation_on_disk(csvfile, thissound, True)

# toggle in frequency
M.ilabel=1
C.toggle_annotation(1)
thissound = M.clustered_sounds[1].copy()
thissound.pop('kind', None)
thissound['label']=M.state['labels'][M.ilabel]
check_annotation_in_memory(thissound, True)
M.save_annotations()
files = os.listdir(basepath)
if len(files)!=2: print("ERROR: wrong number of files")

csvfile = os.path.join(basepath, next(filter(lambda x: x.endswith('.csv'), files)))
count_lines(csvfile, 2)
check_annotation_on_disk(csvfile, thissound, True)

# toggle in ambient
M.ilabel=2
C.toggle_annotation(2)
thissound = M.clustered_sounds[2].copy()
thissound.pop('kind', None)
thissound['label']=M.state['labels'][M.ilabel]
check_annotation_in_memory(thissound, True)
M.save_annotations()
files = os.listdir(basepath)
if len(files)!=2: print("ERROR: wrong number of files")

csvfile = os.path.join(basepath, next(filter(lambda x: x.endswith('.csv'), files)))
count_lines(csvfile, 3)
check_annotation_on_disk(csvfile, thissound, True)

# undo it
C.undo_callback()
check_annotation_in_memory(thissound, False)
M.save_annotations()
files = os.listdir(basepath)
if len(files)!=2: print("ERROR: wrong number of files")

csvfile = os.path.join(basepath, next(filter(lambda x: x.endswith('.csv'), files)))
count_lines(csvfile, 2)
check_annotation_on_disk(csvfile, thissound, False)

# redo it
C.redo_callback()
check_annotation_in_memory(thissound, True)
M.save_annotations()
files = os.listdir(basepath)
if len(files)!=2: print("ERROR: wrong number of files")

csvfile = os.path.join(basepath, next(filter(lambda x: x.endswith('.csv'), files)))
count_lines(csvfile, 3)
check_annotation_on_disk(csvfile, thissound, True)

# toggle out ambient
C.toggle_annotation(2)
check_annotation_in_memory(thissound, False)
M.save_annotations()
files = os.listdir(basepath)
if len(files)!=2: print("ERROR: wrong number of files")

csvfile = os.path.join(basepath, next(filter(lambda x: x.endswith('.csv'), files)))
count_lines(csvfile, 2)
check_annotation_on_disk(csvfile, thissound, False)

# toggle in ambient
M.ilabel=2
C.toggle_annotation(2)
thissound = M.clustered_sounds[2].copy()
thissound.pop('kind', None)
thissound['label']=M.state['labels'][M.ilabel]
check_annotation_in_memory(thissound, True)
M.save_annotations()
files = os.listdir(basepath)
if len(files)!=2: print("ERROR: wrong number of files")

csvfile = os.path.join(basepath, next(filter(lambda x: x.endswith('.csv'), files)))
count_lines(csvfile, 3)
check_annotation_on_disk(csvfile, thissound, True)

# undo all
C.undo_callback()
C.undo_callback()
C.undo_callback()
C.undo_callback()
C.undo_callback()
if len(M.annotated_sounds)!=0: print("ERROR: wrong number of elements")

M.save_annotations()
files = os.listdir(basepath)
if len(files)!=1: print("ERROR: wrong number of files")

##
M.clustered_sounds = [
  {'label': 'pulse', 'file': wavfile, 'ticks': [2, 5], 'kind': 'annotated'},
  {'label': 'sine', 'file': wavfile, 'ticks': [10, 15], 'kind': 'annotated'},
  {'label': 'ambient', 'file': wavfile, 'ticks': [20, 30], 'kind': 'annotated'},
  ]
M.clustered_starts_sorted = [x['ticks'][0] for x in M.clustered_sounds]
M.clustered_stops = [x['ticks'][1] for x in M.clustered_sounds]
M.iclustered_stops_sorted = np.argsort(M.clustered_stops)
csvfile = os.path.join(basepath, 'PS_20130625111709_ch3-annotated-original.csv')
with open(csvfile, 'w') as fid:
  csvwriter = csv.writer(fid)
  for sound in M.clustered_sounds:
    csvwriter.writerow([os.path.basename(sound['file']),
                        sound['ticks'][0], sound['ticks'][1],
                        sound['kind'],
                        sound['label']])

# pulse to ambient
M.ilabel=2
C.toggle_annotation(0)
thissound = M.clustered_sounds[0].copy()
thissound.pop('kind', None)
thissound['label']=M.state['labels'][M.ilabel]
check_annotation_in_memory(thissound, True)
M.save_annotations()
files = os.listdir(basepath)
if len(files)!=3: print("ERROR: wrong number of files")

thissound = M.clustered_sounds[0].copy()
thissound.pop('kind', None)
csvfile = os.path.join(basepath,
                       next(filter(lambda x: x.endswith('.csv') and 'original' in x,
                                   files)))
count_lines(csvfile, 2)
check_annotation_on_disk(csvfile, thissound, False)

thissound['label']=M.state['labels'][M.ilabel]
csvfile = os.path.join(basepath,
                       next(filter(lambda x: x.endswith('.csv') and 'original' not in x,
                                   files)))
count_lines(csvfile, 1)
check_annotation_on_disk(csvfile, thissound, True)

#undo
C.undo_callback()
thissound = M.clustered_sounds[0].copy()
thissound.pop('kind', None)
check_annotation_in_memory(thissound, True)
M.save_annotations()
files = os.listdir(basepath)
if len(files)!=3: print("ERROR: wrong number of files")

csvfile = os.path.join(basepath,
                       next(filter(lambda x: x.endswith('.csv') and 'original' in x,
                                   files)))
count_lines(csvfile, 2)
check_annotation_on_disk(csvfile, thissound, False)

csvfile = os.path.join(basepath,
                       next(filter(lambda x: x.endswith('.csv') and 'original' not in x,
                                   files)))
count_lines(csvfile, 1)
check_annotation_on_disk(csvfile, thissound, True)

#redo
C.redo_callback()
thissound = M.clustered_sounds[0].copy()
thissound.pop('kind', None)
thissound['label']=M.state['labels'][M.ilabel]
check_annotation_in_memory(thissound, True)
M.save_annotations()
files = os.listdir(basepath)
if len(files)!=3: print("ERROR: wrong number of files")

thissound = M.clustered_sounds[0].copy()
thissound.pop('kind', None)
csvfile = os.path.join(basepath,
                       next(filter(lambda x: x.endswith('.csv') and 'original' in x,
                                   files)))
count_lines(csvfile, 2)
check_annotation_on_disk(csvfile, thissound, False)

thissound['label']=M.state['labels'][M.ilabel]
csvfile = os.path.join(basepath,
                       next(filter(lambda x: x.endswith('.csv') and 'original' not in x,
                                   files)))
count_lines(csvfile, 1)
check_annotation_on_disk(csvfile, thissound, True)

# delete sine
M.ilabel=3
C.toggle_annotation(1)
thissound = M.clustered_sounds[1].copy()
thissound.pop('kind', None)
thissound['label']=M.state['labels'][M.ilabel]
check_annotation_in_memory(thissound, True)
M.save_annotations()
files = os.listdir(basepath)
if len(files)!=3: print("ERROR: wrong number of files")

thissound = M.clustered_sounds[1].copy()
thissound.pop('kind', None)
csvfile = os.path.join(basepath,
                       next(filter(lambda x: x.endswith('.csv') and 'original' in x,
                                   files)))
count_lines(csvfile, 1)
check_annotation_on_disk(csvfile, thissound, False)

csvfile = os.path.join(basepath,
                       next(filter(lambda x: x.endswith('.csv') and 'original' not in x,
                                   files)))
count_lines(csvfile, 1)
check_annotation_on_disk(csvfile, thissound, False)

#undo
C.undo_callback()
thissound = M.clustered_sounds[1].copy()
thissound.pop('kind', None)
check_annotation_in_memory(thissound, True)
M.save_annotations()
files = os.listdir(basepath)
if len(files)!=3: print("ERROR: wrong number of files")

csvfile = os.path.join(basepath,
                       next(filter(lambda x: x.endswith('.csv') and 'original' in x,
                                   files)))
count_lines(csvfile, 1)
check_annotation_on_disk(csvfile, thissound, False)

csvfile = os.path.join(basepath,
                       next(filter(lambda x: x.endswith('.csv') and 'original' not in x,
                                   files)))
count_lines(csvfile, 2)
check_annotation_on_disk(csvfile, thissound, True)

#redo
C.redo_callback()
thissound = M.clustered_sounds[1].copy()
thissound.pop('kind', None)
thissound['label']=M.state['labels'][M.ilabel]
check_annotation_in_memory(thissound, True)
M.save_annotations()
files = os.listdir(basepath)
if len(files)!=3: print("ERROR: wrong number of files")

thissound = M.clustered_sounds[1].copy()
thissound.pop('kind', None)
csvfile = os.path.join(basepath,
                       next(filter(lambda x: x.endswith('.csv') and 'original' in x,
                                   files)))
count_lines(csvfile, 1)
check_annotation_on_disk(csvfile, thissound, False)

csvfile = os.path.join(basepath,
                       next(filter(lambda x: x.endswith('.csv') and 'original' not in x,
                                   files)))
count_lines(csvfile, 1)
check_annotation_on_disk(csvfile, thissound, False)

# ambient to pulse
M.ilabel=0
C.toggle_annotation(2)
thissound = M.clustered_sounds[2].copy()
thissound.pop('kind', None)
thissound['label']=M.state['labels'][M.ilabel]
check_annotation_in_memory(thissound, True)
M.save_annotations()
files = os.listdir(basepath)
if len(files)!=2: print("ERROR: wrong number of files")

thissound['label']=M.state['labels'][M.ilabel]
csvfile = os.path.join(basepath,
                       next(filter(lambda x: x.endswith('.csv') and 'original' not in x,
                                   files)))
count_lines(csvfile, 2)
check_annotation_on_disk(csvfile, thissound, True)

#undo
C.undo_callback()
thissound = M.clustered_sounds[2].copy()
thissound.pop('kind', None)
check_annotation_in_memory(thissound, True)
M.save_annotations()
files = os.listdir(basepath)
if len(files)!=2: print("ERROR: wrong number of files")

csvfile = os.path.join(basepath,
                       next(filter(lambda x: x.endswith('.csv') and 'original' not in x,
                                   files)))
count_lines(csvfile, 2)
check_annotation_on_disk(csvfile, thissound, True)

#redo
C.redo_callback()
thissound = M.clustered_sounds[2].copy()
thissound.pop('kind', None)
thissound['label']=M.state['labels'][M.ilabel]
check_annotation_in_memory(thissound, True)
M.save_annotations()
files = os.listdir(basepath)
if len(files)!=2: print("ERROR: wrong number of files")

csvfile = os.path.join(basepath,
                       next(filter(lambda x: x.endswith('.csv') and 'original' not in x,
                                   files)))
count_lines(csvfile, 2)
check_annotation_on_disk(csvfile, thissound, True)
