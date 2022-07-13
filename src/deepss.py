# similar to steinfath, palacios, rottschaefer, yuezak, clemens (2021),
# but with kapre replaced with LEAF (see zeghidour, teboul, quitry,
# tagliasacchi, 2021), and using full 2D convolutions instead of separable
# 1D convolutions.  one can also use spectrograms on multi-channel data

import math

import tensorflow as tf
from tensorflow.keras.layers import *
from leaf_audio.frontend import Leaf
from tcn import TCN

import numpy as np

import logging 
bokehlog = logging.getLogger("songexplorer") 

def _callback(ps,M,V,C):
    C.time.sleep(0.5)
    for p in ps:
        V.model_parameters[p].css_classes = []
    M.save_state_callback()
    V.buttons_update()

def window_callback(n,M,V,C):
    ps = []
    changed, window_ms2 = M.next_pow2_ms(float(V.model_parameters['window_ms'].value))
    if changed:
        V.model_parameters['window_ms'].css_classes = ['changed']
        V.model_parameters['window_ms'].value = str(window_ms2)
        ps.append('window_ms')
    if changed:
        if V.bokeh_document:
            V.bokeh_document.add_next_tick_callback(lambda: _callback(ps,M,V,C))
        else:
            _callback(ps,M,V,C)

def stride_callback(n,M,V,C):
    if V.model_parameters['representation'].value == "waveform":
        stride_ms = 1000/M.audio_tic_rate
    else:
        stride_ms = float(V.model_parameters['stride_ms'].value)
    output_tic_rate = 1000 / stride_ms
    if output_tic_rate != round(output_tic_rate):
        stride_ms2 = 1000 / math.floor(output_tic_rate)
        V.model_parameters['stride_ms'].css_classes = ['changed']
        V.model_parameters['stride_ms'].value = str(stride_ms2)
        if V.bokeh_document:
            V.bokeh_document.add_next_tick_callback(lambda: _callback(['stride_ms'],M,V,C))
        else:
            _callback(['stride_ms'],M,V,C)

use_audio=1
use_video=0

model_parameters = [
  # key, title in GUI, '' for textbox or [] for pull-down, default value, enable logic, callback, required
  ["representation",     "representation", ['waveform',
                                            'leaf'],         'leaf',       [],                None,            True],
  ["window_ms",          "window (msec)",  '',               '6.4',        ["representation",
                                                                            ["leaf"]],        window_callback, True],
  ["stride_ms",          "stride (msec)",  '',               '1.6',        ["representation",
                                                                            ["leaf"]],        stride_callback, True],
  ["nfilters",           "# filters",      '',               '9',          ["representation",
                                                                            ["leaf"]],        None,            True],
  ["kernel_size",        "kernel size",    '',               '8',          [],                None,            True],
  ["nstacks",            "# stacks",       '',               '3',          [],                None,            True],
  ["nfeatures",          "# features",     '',               '16',         [],                None,            True],
  ["dilations",          "dilations",      '',               '1,2,4,8,16', [],                None,            True],
  ["upsample",           "upsample",       ['yes','no'],     'yes',        [],                None,            True],
  ["connection_type",    "connection",     ['plain',
                                            'skip'],         'skip',       [],                None,            True],
  ["dropout",            "dropout",        '',               '0.1',        [],                None,            True],
  ]

class Slice(tf.keras.layers.Layer):
    def __init__(self, begin, size, **kwargs):
        super(Slice, self).__init__(**kwargs)
        self.begin = begin
        self.size = size
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'begin': self.begin,
            'size': self.size,
        })
        return config
    def call(self, inputs):
        return tf.slice(inputs, self.begin, self.size)

def create_model(model_settings, model_parameters):
    nchannels = model_settings['nchannels']
    parallelize = model_settings['parallelize']
    context_tics = int(model_settings['audio_tic_rate'] * model_settings['context_ms'] / 1000)
    audio_tic_rate = model_settings['audio_tic_rate']
    representation = model_parameters['representation']
    kernel_size = int(model_parameters['kernel_size'])
    nstacks = int(model_parameters['nstacks'])
    dilations = [int(x) for x in model_parameters['dilations'].split(',')]

    if representation == "waveform":
        window_tics = stride_tics = 1
    else:
        window_tics = round(audio_tic_rate * float(model_parameters['window_ms']) / 1000)
        stride_tics = round(audio_tic_rate * float(model_parameters['stride_ms']) / 1000)

        if not (window_tics & (window_tics-1) == 0) or window_tics == 0:
            next_higher = np.power(2, np.ceil(np.log2(window_tics))).astype(np.int)
            next_lower = np.power(2, np.floor(np.log2(window_tics))).astype(np.int)
            sigdigs = np.ceil(np.log10(next_higher)).astype(np.int)+1
            next_higher_ms = np.around(next_higher/audio_tic_rate*1000, decimals=sigdigs)
            next_lower_ms = np.around(next_lower/audio_tic_rate*1000, decimals=sigdigs)
            raise Exception("ERROR: 'window (msec)' should be a power of two when converted to tics.  "+
                             model_parameters['window_ms']+" ms is "+str(window_tics)+
                             " tics for Fs="+str(audio_tic_rate)+".  try "+str(next_lower_ms)+
                             " ms (="+str(next_lower)+") or "+str(next_higher_ms)+"ms (="+
                             str(next_higher)+") instead.")

    receptive_field_tics = stride_tics * (1 + 2 * (kernel_size - 1) * nstacks * sum(dilations))
    print("receptive_field_tics = %d" % receptive_field_tics)

    output_tic_rate = 1000 / float(model_parameters['stride_ms'])
    if output_tic_rate != round(output_tic_rate):
        raise Exception("ERROR: 1000 / 'stride (msec)' should be an integer")
    else:
      print('output_tic_rate = '+str(output_tic_rate))

    hidden_layers = []

    ninput_tics = context_tics + (parallelize-1) * stride_tics

    input_layer = Input(shape=(ninput_tics, nchannels))
    hidden_layers.append(input_layer)
    x = input_layer

    if representation == "leaf":
        spectrograms = []
        for ichan in range(nchannels):
            chan = Slice([0, 0, ichan], [-1,-1,1])(x)
            spectrograms.append(Leaf(n_filters=int(model_parameters['nfilters']),
                                     window_len=window_tics,
                                     window_stride=stride_tics,
                                     sample_rate=audio_tic_rate,
                                     name='leaf'+str(ichan))(chan))
        x = Concatenate()(spectrograms) if nchannels>1 else spectrograms[0]
        hidden_layers.append(x)

    tcn = TCN(nb_filters=int(model_parameters['nfeatures']),
              kernel_size=kernel_size,
              nb_stacks=nstacks,
              dilations=dilations,
              return_sequences=True,
              use_skip_connections=model_parameters['connection_type']=='skip',
              dropout_rate=float(model_parameters['dropout']))
    tcn.build(x.shape)

    # recapitulate tcn.call() to capture hidden layers
    skip_connections = []
    for layer in tcn.residual_blocks:
        x, skip_out = layer(x)
        skip_connections.append(skip_out)
        hidden_layers.append(x)
    if tcn.use_skip_connections:
        x = Add()(skip_connections)
        hidden_layers.append(x)

    x = Slice([ 0, x.shape[1]-parallelize,  0],
              [-1, parallelize,            -1])(x)
    hidden_layers.append(x)

    x = Conv1D(model_settings['nlabels'], 1)(x)
    if representation == "leaf" and model_parameters['upsample']=='yes':
        x = UpSampling1D(size=stride_tics)(x)
    output_layer = x

    return tf.keras.Model(inputs=input_layer, outputs=[hidden_layers, output_layer])
