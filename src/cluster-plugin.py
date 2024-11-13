#!/usr/bin/env python

# e.g. cluster-plugin.py \
#     --data_dir=`pwd`/groundtruth-data \
#     --layers=0,1,2,3,4 \
#     --parameters={"my-simple-textbox":"16", "a-bounded-value":"1", "a-menu":"that", "a-conditional-param":"6", "an-optional-param":""}
 
import argparse
import os
import sys
from datetime import datetime
import socket
from natsort import natsorted
from random import random
import json
import numpy as np

# optional callbacks can be used to validate user input
def _callback(p,M,V,C):
    C.time.sleep(0.5)
    V.cluster_parameters[p].css_classes = []
    M.save_state_callback()
    V.buttons_update()

def callback(n,M,V,C):
    # M, V, C are the model, view, and controller in src/gui
    # access the hyperparameters below with the V.detect_parameters dictionary
    # the value is stored in .value, and the appearance can be controlled with .css_classes
    if int(V.detect_parameters['a-bounded-value'].value) < 0:
        #bokehlog.info("a-bounded-value = "+str(V.detect_parameters['a-bounded-value'].value))  # uncomment to debug
        V.detect_parameters['a-bounded-value'].css_classes = ['changed']
        V.detect_parameters['a-bounded-value'].value = "0"
        if V.bokeh_document:  # if interactive V.bokeh_document.add_next_tick_callback(lambda: _callback('a-bounded-value',M,V,C))
            V.bokeh_document.add_next_tick_callback(lambda: _callback('a-bounded-value',M,V,C))
        else:  # if scripted
            _callback('a-bounded-value',M,V,C)

# a list of lists specifying the cluster-specific hyperparameters in the GUI
def cluster_parameters():
    return [
        # [key in `cluster_parameters`, title in GUI, "" for textbox or [] for pull-down, default value, width, enable logic, callback, required]
          ["my-simple-textbox",    "h-parameter 1",    "",              "32",   1, [],                  None,     True],
          ["a-bounded-value",      "can't be < 0",     "",              "3",    1, [],                  callback, True],
          ["a-menu",               "choose one",       ["this","that"], "this", 1, [],                  None,     True],
          ["a-conditional-param",  "that's parameter", "",              "8",    1, ["a-menu",["that"]], None,     True],
          ["an-optional-param",    "can be blank",     "",              "0.5",  1, [],                  None,     False],
    ]

# a script which inputs activations.npz and outputs cluster.npz
def main():
  flags = vars(FLAGS)
  for key in sorted(flags.keys()):
    print('%s = %s' % (key, flags[key]))

  layers = [int(x) for x in FLAGS.layers.split(',')]

  print("loading data...")
  activations=[]
  npzfile = np.load(os.path.join(FLAGS.data_dir, 'activations.npz'),
                    allow_pickle=True)
  sounds = npzfile['sounds']
  for arr_ in natsorted(filter(lambda x: x.startswith('arr_'), npzfile.files)):
    activations.append(npzfile[arr_])

  nlayers = len(activations)

  activations_flattened = [None]*nlayers
  for ilayer in layers:
    nsounds = np.shape(activations[ilayer])[0]
    activations_flattened[ilayer] = np.reshape(activations[ilayer],(nsounds,-1))
    print("length of layer "+str(ilayer)+" is "+str(np.shape(activations_flattened[ilayer])))

  activations_clustered = [None]*nlayers
  for ilayer in layers:
    print("reducing dimensionality of layer "+str(ilayer))
    activations_clustered[ilayer] = np.array([[random(),random()]
                                              for _ in range(np.shape(activations_flattened[ilayer])[0])])

  np.savez(os.path.join(FLAGS.data_dir, 'cluster'), \
           sounds = sounds,
           activations_clustered = np.array(activations_clustered, dtype=object),
           labels_touse = npzfile['labels_touse'],
           kinds_touse = npzfile['kinds_touse'])
  
if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--data_dir',
      type=str)
  parser.add_argument(
      '--layers',
      type=str)
  parser.add_argument(
      '--parallelize',
      type=int,
      default=0)
  parser.add_argument(
      '--parameters',
      type=json.loads,
      default='{"my-simple-textbox":16, "a-bounded-value":1, "a-menu":"that", "a-conditional-param":6, "an-optional-param":""}')

  FLAGS, unparsed = parser.parse_known_args()

  print(str(datetime.now())+": start time")
  repodir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
  with open(os.path.join(repodir, "VERSION.txt"), 'r') as fid:
    print('SongExplorer version = '+fid.read().strip().replace('\n',', '))
  print("hostname = "+socket.gethostname())
  
  try:
    main()

  except Exception as e:
    print(e)
  
  finally:
    if hasattr(os, 'sync'):
      os.sync()
    print(str(datetime.now())+": finish time")
