# an architecture plugin for transfer learning.  pretrained models can have
# different audio tic rates, but the context window and strides for the
# spectrogram and convolutional layers must be the same.

import os
import sys
import re
import importlib
import json

import tensorflow as tf
from tensorflow.keras.layers import *
import tensorflow_io as tfio

from lib import get_srcrepobindirs

import logging 
bokehlog = logging.getLogger("songexplorer") 

use_audio=1
use_video=0

model_parameters = [
    # key, title in GUI, '' for textbox or [] for pull-down, default value, width, enable logic, callback, required
    ["ckpt_files",    "checkpoint file(s)", '', '',  6, [], None, True],
    ["trainable",     "trainable?",         '', '',  1, [], None, False],
    ["splice_layers", "layer(s)",           '', '',  1, [], None, False],
    ["conv_layers",   "conv",               '', '',  2, [], None, False],
    ["dense_layers",  "dense",              '', '',  1, [], None, False],
    ["dropout_rate",  "dropout %",          '', '0', 1, [], None, True],
    ]

def load_model(context_ms0, audio_tic_rate0, parallelize0, ckpt_file, io):
    logfile = os.path.dirname(ckpt_file)+".log"
    with open(logfile, "r") as fid:
        for line in fid:
            if "audio_tic_rate = " in line:
                m=re.search('audio_tic_rate = (.*)', line)
                audio_tic_rate = int(m.group(1))
            elif "audio_nchannels = " in line:
                m=re.search('audio_nchannels = (.*)', line)
                audio_nchannels = int(m.group(1))
            elif "video_frame_rate = " in line:
                m=re.search('video_frame_rate = (.*)', line)
                video_frame_rate = int(m.group(1))
            elif "video_frame_width = " in line:
                m=re.search('video_frame_width = (.*)', line)
                video_frame_width = int(m.group(1))
            elif "video_frame_height = " in line:
                m=re.search('video_frame_height = (.*)', line)
                video_frame_height = int(m.group(1))
            elif "video_channels = " in line:
                m=re.search('video_channels = (.*)', line)
                video_channels = m.group(1)
            elif "context_ms = " in line:
                m=re.search('context_ms = (.*)', line)
                context_ms = float(m.group(1))
            elif "labels_touse = " in line:
                m=re.search('labels_touse = (.*)', line)
                nlabels = len(m.group(1).split(','))
            elif "model_architecture = " in line:
                m=re.search('model_architecture = (.*)', line)
                model_architecture = m.group(1)
            elif "model_parameters = " in line:
                m=re.search('model_parameters = (.*)', line)
                _model_parameters = json.loads(m.group(1).replace("'",'"'))
            elif "num validation labels" in line:
                break

    if context_ms0 != context_ms:
        raise Exception(f"ERROR:  context_ms of {ckpt_file} is {context_ms} "
                        f"whereas it is currently {context_ms0} in the settings.")

    if audio_tic_rate0 != audio_tic_rate:
        print(f"INFO:  audio_tic_rate of {ckpt_file} is {audio_tic_rate} "
              f"whereas it is currently {audio_tic_rate0} in the settings.  "
              f"will resample accordingly.", file=io)

    model_settings = {'nlabels': nlabels,
                      'audio_tic_rate': audio_tic_rate,
                      'audio_nchannels': audio_nchannels,
                      'video_frame_rate': video_frame_rate,
                      'video_frame_width': video_frame_width,
                      'video_frame_height': video_frame_height,
                      'video_channels': video_channels,
                      'parallelize': parallelize0,
                      'batch_size': 1,
                      'context_ms': context_ms}  ### ???

    srcdir, _, _ = get_srcrepobindirs()
    modeldir = os.path.dirname(model_architecture)
    sys.path.append(modeldir if modeldir else srcdir)
    model = importlib.import_module(os.path.basename(model_architecture))

    thismodel = model.create_model(model_settings, _model_parameters)

    checkpoint = tf.train.Checkpoint(thismodel=thismodel)
    checkpoint.read(ckpt_file).expect_partial()

    return thismodel, audio_tic_rate

def strip_ckpt(ckpt_file):
    basename = os.path.basename(ckpt_file)
    if not os.path.exists(ckpt_file):
        raise Exception(f"ERROR:f {ckpt_file} does not exist")
    elif basename.endswith('.index'):
        ckpt_file= ckpt_file[:-6]
    elif '.data' in basename:
        ckpt_file= ckpt_file[:ckpt_file.rindex('.data')]
    else:
        raise Exception(f"ERROR:  {ckpt_file} is not a checkpoint file")
    return ckpt_file

class Resample(tf.keras.layers.Layer):
    def __init__(self, rate_in, rate_out, **kwargs):
        super(Resample, self).__init__(**kwargs)
        self.rate_in = rate_in
        self.rate_out = rate_out
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'rate_in': self.rate_in,
            'rate_out': self.rate_out,
        })
        return config
    def call(self, inputs):
        return tfio.audio.resample(inputs, rate_in=self.rate_in, rate_out=self.rate_out)

def create_model(model_settings, model_parameters, io=sys.stdout):
    ckpt_files = []
    for ckpt_file in model_parameters['ckpt_files'].split(','):
        ckpt_files.append(strip_ckpt(ckpt_file))
    trainable = [0]*len(ckpt_files) if model_parameters['trainable']=='' \
                else [int(x) for x in model_parameters['trainable'].split(',')]
    splice_layers = [-1]*len(ckpt_files) if model_parameters['splice_layers']=='' \
                    else [int(x) for x in model_parameters['splice_layers'].split(',')]
    conv_layers = [] if model_parameters['conv_layers']=='' \
                  else [[int(y) for y in x.split('x')]
                        for x in model_parameters['conv_layers'].split(',')]
    dense_layers = [] if model_parameters['dense_layers']=='' \
                   else [int(x) for x in model_parameters['dense_layers'].split(',')]
    dropout_rate = float(model_parameters['dropout_rate'])/100

    hidden_layers = []

    models = []
    audio_tic_rates = []
    ninput_tics = None
    for i in range(len(ckpt_files)):
        model, audio_tic_rate = load_model(model_settings['context_ms'],
                                           model_settings['audio_tic_rate'],
                                           model_settings['parallelize'],
                                           ckpt_files[i], io)
        if ninput_tics:
            if model.input.shape[1] != ninput_tics:
                raise Exception(f"ERROR: # input samples for {ckpt_files[i]} is {m.input.shape[1]} "
                                f"whereas for {ckpt_files[0]} it is {ninput_tics}, "
                                f"possibly due to different spectrogram / convolution strides.  "
                                f"try setting parallelize to 1")
        else:
            ninput_tics = model.input.shape[1]
        model.trainable = trainable[i]
        truncated_model = tf.keras.Model(inputs=model.inputs,
                                         outputs=[model.output[0][1:splice_layers[i]],
                                                  model.output[0][splice_layers[i]]])
        models.append(truncated_model)
        audio_tic_rates.append(audio_tic_rate)

    ninput_tics = round(models[0].input.shape[1] / audio_tic_rates[0] *
                        model_settings['audio_tic_rate'])
    inputs = Input(shape=(ninput_tics, model_settings['audio_nchannels']))
    lowerlegs = []
    for (m,fs) in zip(models, audio_tic_rates):
        if fs==model_settings['audio_tic_rate']:
            x = Identity()
        else:
            l = Resample(model_settings['audio_tic_rate'], fs)
        x = l(inputs)
        x = m(x)
        hidden_layers.extend(x[0])
        lowerlegs.append(tf.keras.Model(inputs=inputs, outputs=x[1]))

    upperlegs = []
    for leg in lowerlegs:
        x = leg.output
        for (t,f,m) in conv_layers:
            x = ReLU()(x)
            x = Conv2D(m, (t,f))(x)
            hidden_layers.append(x)
        upperlegs.append(x)

    x = Concatenate()([Reshape((1,-1))(x) for x in upperlegs])

    for idense, nunits in enumerate(dense_layers+[model_settings['nlabels']]):
        if idense>0:
            x = ReLU()(x)
        if dropout_rate>0:
            x = Dropout(dropout_rate)(x)
        x = Conv1D(nunits, 1)(x)
        hidden_layers.append(x)
    
    final = Reshape((-1,model_settings['nlabels']))(x)

    print('ensemble-transfer.py version = 0.1', file=io)
    return tf.keras.Model(inputs=inputs, outputs=[hidden_layers, final], name="ensemble-transfer")
