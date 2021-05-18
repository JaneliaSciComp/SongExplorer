# similar to steinfath, palacios, rottschaefer, yuezak, clemens (2021),
# but with kapre replaced with LEAF (see zeghidour, teboul, quitry,
# tagliasacchi, 2021), and using full 2D convolutions instead of separable
# 1D convolutions.  one can also use spectrograms on multi-channel data

import tensorflow as tf
from tensorflow.keras.layers import *
from leaf_audio.frontend import Leaf
from tcn import TCN

model_parameters = [
  # key in `model_settings`, title in GUI, '' for textbox or [] for pull-down, default value, enable logic
  ["representation",     "representation", ['waveform',
                                            'leaf'],         'leaf',       []],
  ["window_ms",          "window (msec)",  '',               '6.4',        ["representation",
                                                                            ["leaf"]]],
  ["stride_ms",          "stride (msec)",  '',               '1.6',        ["representation",
                                                                            ["leaf"]]],
  ["nfilters",           "# filters",      '',               '9',          ["representation",
                                                                            ["leaf"]]],
  ["kernel_size",        "kernel size",    '',               '8',          []],
  ["nstacks",            "# stacks",       '',               '3',          []],
  ["nfeatures",          "# features",     '',               '16',         []],
  ["dilations",          "dilations",      '',               '1,2,4,8,16', []],
  ["upsample",           "upsample",       ['yes','no'],     'yes',        []],
  ["connection_type",    "connection",     ['plain',
                                            'skip'],         'skip',       []],
  ["dropout",            "dropout",        '',               '0.1',        []],
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

#`model_settings` is a dictionary of hyperparameters
def create_model(model_settings):
    nchannels = model_settings['nchannels']
    parallelize = model_settings['parallelize']
    context_tics = model_settings['context_tics']
    audio_tic_rate = model_settings['audio_tic_rate']
    representation = model_settings['representation']
    if representation == "waveform":
        window_tics = stride_tics = 1
    else:
        window_tics = int(audio_tic_rate * float(model_settings['window_ms']) / 1000)
        stride_tics = int(audio_tic_rate * float(model_settings['stride_ms']) / 1000)

    kernel_size = int(model_settings['kernel_size'])
    nstacks = int(model_settings['nstacks'])
    dilations = [int(x) for x in model_settings['dilations'].split(',')]
    receptive_field_tics = stride_tics * (1 + 2 * (kernel_size - 1) * nstacks * sum(dilations))
    print("receptive_field_tics = %d" % receptive_field_tics)

    hidden_layers = []

    ninput_tics = context_tics + (parallelize-1) * stride_tics

    input_layer = Input(shape=(ninput_tics, nchannels))
    hidden_layers.append(input_layer)
    x = input_layer

    if representation == "leaf":
        spectrograms = []
        for ichan in range(nchannels):
            chan = Slice([0, 0, ichan], [-1,-1,1])(x)
            spectrograms.append(Leaf(n_filters=int(model_settings['nfilters']),
                                     window_len=window_tics,
                                     window_stride=stride_tics,
                                     sample_rate=audio_tic_rate,
                                     name='leaf'+str(ichan))(chan))
        x = Concatenate()(spectrograms) if nchannels>1 else spectrograms[0]
        hidden_layers.append(x)

    tcn = TCN(nb_filters=int(model_settings['nfeatures']),
              kernel_size=kernel_size,
              nb_stacks=nstacks,
              dilations=dilations,
              return_sequences=True,
              use_skip_connections=model_settings['connection_type']=='skip',
              dropout_rate=float(model_settings['dropout']))
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
    if representation == "leaf" and model_settings['upsample']=='yes':
        x = UpSampling1D(size=stride_tics)(x)
    output_layer = x

    return tf.keras.Model(inputs=input_layer, outputs=[hidden_layers, output_layer])
