#!/usr/bin/env python3

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

# a script which inputs a WAV file and outputs a CSV file
if __name__ == '__main__':

    import os
    import scipy.io.wavfile as spiowav
    import sys
    import csv
    import socket
    import json
    from datetime import datetime
    import shutil

    repodir = os.path.dirname(os.path.dirname(os.path.realpath(shutil.which("songexplorer"))))

    print(str(datetime.now())+": start time")
    with open(os.path.join(repodir, "VERSION.txt"), 'r') as fid:
      print('SongExplorer version = '+fid.read().strip().replace('\n',', '))
    print("hostname = "+socket.gethostname())

    try:

        _, filename, detect_parameters, audio_tic_rate, audio_nchannels = sys.argv
        print('filename: '+filename)
        print('detect_parameters: '+detect_parameters)
        print('audio_tic_rate: '+audio_tic_rate)
        print('audio_nchannels: '+audio_nchannels)

        detect_parameters = json.loads(detect_parameters)
        audio_tic_rate = int(audio_tic_rate)
        audio_nchannels = int(audio_nchannels)

        hyperparameter1 = int(detect_parameters["my-simple-textbox"])

        _, song = spiowav.read(filename)
        song = abs(song)
        if audio_nchannels>1:
            song = np.max(song, axis=1)
        amplitude = scipy.signal.medfilt(song)

        basename = os.path.basename(filename)
        with open(os.path.splitext(filename)[0]+'-detected.csv', 'w') as fid:
            csvwriter = csv.writer(fid)
            for i in range(1,len(amplitude)-1):
                if amplitude[i] > hyperparameter1:
                    if amplitude[i-1] < hyperparameter1):
                        csvwriter.writerow([basename,i,i,'detected','onset'])
                    if amplitude[i+1] < hyperparameter1):
                        csvwriter.writerow([basename,i,i,'detected','offset'])

    except Exception as e:
        print(e)

    finally:
        os.sync()
        print(str(datetime.now())+": finish time")
