#!/usr/bin/env python3

# e.g.
# $SONGEXPLORER_BIN detect-plugin.py \
#     --filename=`pwd`/groundtruth-data/round2/20161207T102314_ch1_p1.wav \
#     --parameters={"time_sigma":"9,4", "time_smooth_ms":"6.4", "frequency_n_ms":"25.6", "frequency_nw":"4", "frequency_p":"0.1,1.0", "frequency_smooth_ms":"25.6", "time_sigma_robust":"median"} \
#     --audio_tic_rate=2500 \
#     --audio_nchannels=1 \
#     --audio_read_plugin=load-wav \
#     --audio_read_plugin_kwargs={}
 
import argparse
import os
import importlib
import sys
import csv
import socket
import json
from datetime import datetime
import shutil

# use bokehlog.info() to print debugging messages
import logging 
bokehlog = logging.getLogger("songexplorer") 

# optional callbacks can be used to validate user input
def _callback(p,M,V,C):
    C.time.sleep(0.5)
    V.model_parameters[p].css_classes = []
    M.save_state_callback()
    V.buttons_update()

def callback(n,M,V,C):
    # M, V, C are the model, view, and controller in src/gui
    # access the hyperparameters below with the V.model_parameters dictionary
    # the value is stored in .value, and the appearance can be controlled with .css_classes
    if int(V.model_parameters['a-bounded-value'].value) < 0:
        #bokehlog.info("a-bounded-value = "+str(V.model_parameters['a-bounded-value'].value))  # uncomment to debug
        V.model_parameters['a-bounded-value'].css_classes = ['changed']
        V.model_parameters['a-bounded-value'].value = "0"
        if V.bokeh_document:  # if interactive
            V.bokeh_document.add_next_tick_callback(lambda: _callback('a-bounded-value',M,V,C))
        else:  # if scripted
            _callback('a-bounded-value',M,V,C)

# a list of lists specifying the detect-specific hyperparameters in the GUI
detect_parameters = [
  # [key in `detect_parameters`, title in GUI, "" for textbox or [] for pull-down, default value, width, enable logic, callback, required]
  ["my-simple-textbox",    "h-parameter 1",    "",              "32",   1, [],                  None,     True],
  ["a-bounded-value",      "can't be < 0",     "",              "3",    1, [],                  callback, True],
  ["a-menu",               "choose one",       ["this","that"], "this", 1, [],                  None,     True],
  ["a-conditional-param",  "that's parameter", "",              "8",    1, ["a-menu",["that"]], None,     True],
  ["an-optional-param",    "can be blank",     "",              "0.5",  1, [],                  None,     False],
  ]

# a function which returns a vector of strings used to annotate the detected events
def detect_labels(audio_nchannels):
    return ['onset', 'offset']

FLAGS = None

# a script which inputs a WAV file and outputs a CSV file
def main():
  flags = vars(FLAGS)
  for key in sorted(flags.keys()):
    print('%s = %s' % (key, flags[key]))

  sys.path.append(os.path.dirname(FLAGS.audio_read_plugin))
  audio_read_module = importlib.import_module(os.path.basename(FLAGS.audio_read_plugin))
  def audio_read(wav_path, start_tic=None, stop_tic=None):
      return audio_read_module.audio_read(wav_path, start_tic, stop_tic,
                                          **FLAGS.audio_read_plugin_kwargs)

  hyperparameter1 = int(FLAGS.parameters["my-simple-textbox"])

  _, song = audio_read(FLAGS.filename)
  song = abs(song)
  if FLAGS.audio_nchannels>1:
      song = np.max(song, axis=1)
  amplitude = scipy.signal.medfilt(song)

  basename = os.path.basename(FLAGS.filename)
  with open(os.path.splitext(FLAGS.filename)[0]+'-detected.csv', 'w') as fid:
      csvwriter = csv.writer(fid)
      for i in range(1,len(amplitude)-1):
          if amplitude[i] > hyperparameter1:
              if amplitude[i-1] < hyperparameter1):
                  csvwriter.writerow([basename,i,i,'detected','onset'])
              if amplitude[i+1] < hyperparameter1):
                  csvwriter.writerow([basename,i,i,'detected','offset'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--filename',
        type=str)
    parser.add_argument(
        '--parameters',
        type=json.loads)
    parser.add_argument(
        '--audio_tic_rate',
        type=int)
    parser.add_argument(
        '--audio_nchannels',
        type=int)
    parser.add_argument(
        '--audio_read_plugin',
        type=str)
    parser.add_argument(
        '--audio_read_plugin_kwargs',
        type=json.loads)
  
    FLAGS, unparsed = parser.parse_known_args()

    print(str(datetime.now())+": start time")
    print('detect-plugin.py version = 0.1')
    print("hostname = "+socket.gethostname())

    try:
        main()

    except Exception as e:
        print(e)

    finally:
        if hasattr(os, 'sync'):
            os.sync()
        print(str(datetime.now())+": finish time")
