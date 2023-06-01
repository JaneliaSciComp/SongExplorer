import os
import sys
import re
import importlib
import json

import tensorflow as tf
from tensorflow.keras.layers import *

import logging 
bokehlog = logging.getLogger("songexplorer") 

use_audio=1
use_video=1

model_parameters = [
    # key, title in GUI, '' for textbox or [] for pull-down, default value, width, enable logic, callback, required
    ["audio_ckpt",   "audio checkpoint file", '',          '',         6, [], None, True],
    ["video_ckpt",   "video checkpoint file", '',          '',         6, [], None, True],
    ["trainable",    "trainable",             ['audio',
                                               'video',
                                               'both',
                                               'neither'], 'neither',  1, [], None, False],
    ["dense_layers", "dense",                 '',          '',         1, [], None, False],
    ["dropout_rate", "dropout %",             '',          '0',        1, [], None, True],
    ]

def load_model(model_settings, ckpt):
    logfile = os.path.dirname(ckpt)+".log"
    with open(logfile, "r") as fid:
        for line in fid:
            if "model_architecture = " in line:
                m=re.search('model_architecture = (.*)', line)
                this_model_architecture = m.group(1)
            elif "model_parameters = " in line:
                m=re.search('model_parameters = (.*)', line)
                this_model_parameters = json.loads(m.group(1))
                break

    sys.path.append(os.path.dirname("songexplorer/src/"+this_model_architecture))
    model = importlib.import_module(os.path.basename(this_model_architecture))

    thismodel = model.create_model(model_settings, this_model_parameters)

    checkpoint = tf.train.Checkpoint(thismodel=thismodel)
    checkpoint.read(ckpt).expect_partial()

    return thismodel

def strip_ckpt(ckpt):
    basename = os.path.basename(ckpt)
    if not os.path.exists(ckpt):
      print("ERROR: "+ckpt+" does not exist")
    elif basename.endswith('.index'):
      ckpt = ckpt[:-6]
    elif '.data' in basename:
      ckpt = ckpt[:ckpt.rindex('.data')]
    else:
      print("ERROR: "+ckpt+" is not a checkpoint file")
    return ckpt

def create_model(model_settings, model_parameters):
    audio_ckpt = strip_ckpt(model_parameters['audio_ckpt'])
    video_ckpt = strip_ckpt(model_parameters['video_ckpt'])
    trainable = model_parameters['trainable']
    dense_layers = [] if model_parameters['dense_layers']=='' \
                   else [int(x) for x in model_parameters['dense_layers'].split(',')]
    dropout_rate = float(model_parameters['dropout_rate'])/100

    hidden_layers = []

    audio_model = load_model(model_settings, audio_ckpt)
    hidden_layers.extend(audio_model.output[0])
    hidden_layers.append(audio_model.output[1])
    audio_model.trainable = trainable in ['audio', 'both']

    video_model = load_model(model_settings, video_ckpt)
    hidden_layers.extend(video_model.output[0])
    hidden_layers.append(video_model.output[1])
    video_model.trainable = trainable in ['video', 'both']

    x = Concatenate(axis=2)([tf.nn.softmax(audio_model.output[1]),
                             tf.nn.softmax(video_model.output[1])])
    for idense, nunits in enumerate(dense_layers+[model_settings['nlabels']]):
        if idense>0:
            x = ReLU()(x)
        if dropout_rate>0:
            x = Dropout(dropout_rate)(x)
        x = Conv1D(nunits, 1)(x)
        hidden_layers.append(x)
    
    print('ensemble-concat-dense.py version = 0.1')
    return tf.keras.Model(inputs=[audio_model.input, video_model.input],
                          outputs=[hidden_layers, x],
                          name="ensemble-concat-dense")
