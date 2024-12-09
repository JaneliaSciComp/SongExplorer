import sys
import math

# all imported packages must be in the container
import tensorflow as tf
from tensorflow.keras.layers import *

# use bokehlog.info() to print debugging messages
import logging 
bokehlog = logging.getLogger("songexplorer") 

use_audio=1
use_video=0

# a list of lists specifying the architecture-specific hyperparameters in the GUI
def model_parameters(time_units, freq_units, time_scale, freq_scale):
    return [
          # [key, title in GUI, "" for textbox or [] for pull-down, default value, width, enable logic, callback, required]
          ["ndown",           "# down blocks",           '',               '4',                    1, [],                 None,             True],
          ["nup",             "# up blocks",             '',               '4',                    1, [],                 None,             True],
          ["nconv",           "# conv/block",            '',               '2',                    1, [],                 None,             True],
          ["nfilters",        "# filters",               '',               '64',                   1, [],                 None,             True],
          ["kernel_size",     "kernel size",             '',               '3',                    1, [],                 None,             True],
          ["downsample",      "downsample",              ["pool",
                                                          "conv"],         'pool',                 1, [],                 None,             True],
          ["padding",         "padding",                 ["same",
                                                          "valid"],        'valid',                1, [],                 None,             True],
          ["dropout",         "dropout %",               '',               '50',                   1, [],                 None,             True],
          ["output",          "output",                  ["slice",
                                                          "conv"],         'conv',                 1, [],                 None,             True],
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

# a function which returns a keras model
def create_model(model_settings, model_parameters, io=sys.stdout):
    # `model_settings` is a dictionary of additional hyperparameters
    audio_tic_rate = model_settings['audio_tic_rate']
    time_units = model_settings['time_units']
    time_scale = model_settings['time_scale']
    ndown = int(model_parameters['ndown'])
    nup = int(model_parameters['nup'])
    nconv = int(model_parameters['nconv'])
    nfilters = int(model_parameters['nfilters'])
    kernel_size = int(model_parameters['kernel_size'])
    padding = model_parameters['padding']
    dropout = float(model_parameters['dropout'])/100
    #hyperparameter1 = int(model_parameters["my-simple-textbox"])
    #nonnegative = int(model_parameters["a-bounded-value"])

    window_tics = stride_tics = 1

    downsample_by = 2 ** (ndown-nup)
    output_tic_rate = audio_tic_rate / stride_tics / downsample_by
    print('downsample_by = '+str(downsample_by), file=io)
    print('output_tic_rate = '+str(output_tic_rate), file=io)

    # hidden_layers is used to visualize intermediate clusters in the GUI
    hidden_layers = []

    # 'parallelize' specifies the number of output tics to evaluate
    # simultaneously when classifying.  stride (from e.g. spectrograms)
    # and downsampling (from e.g. conv kernel strides) must be taken into
    # account to get the corresponding number of input tics
    context_tics = int(audio_tic_rate * model_settings['context'] * time_scale)
    ninput_tics = context_tics + (model_settings['parallelize']-1) * stride_tics * downsample_by
    noutput_tics = (context_tics-window_tics) // stride_tics + 1
    input_layer = Input(shape=(ninput_tics, model_settings["audio_nchannels"]))
    hidden_layers.append(input_layer)
    x = input_layer

    receptive_field = 1
    sum_of_strides = 0

    # adapted from https://github.com/krentzd/sparse-unet/blob/main/model.py
    # to make it look like https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/
    def down_block_1D(x, nfilters):
        nonlocal noutput_tics, receptive_field 
        for _ in range(nconv):
            x = Conv1D(filters=nfilters, kernel_size=kernel_size, padding=padding)(x)
            receptive_field += (kernel_size-1) * 2**sum_of_strides
            hidden_layers.append(x)
            x = BatchNormalization()(x)
            x = ReLU()(x)
            if padding == 'valid':
                noutput_tics = noutput_tics - kernel_size + 1

        return x

    def up_block_1D(x, concat_layer, nfilters):
        nonlocal noutput_tics, receptive_field, sum_of_strides 
        x = Conv1DTranspose(nfilters, kernel_size=2, strides=2)(x)
        receptive_field += (2-1) * 2**sum_of_strides
        sum_of_strides += 2 - 1
        noutput_tics = math.ceil(noutput_tics * 2 - 1 + 2 - 1)   ### ???

        x_shape = x.shape
        concat_shape = concat_layer.shape
        hoffset = (concat_shape[1] - x_shape[1]) // 2
        x = Concatenate()([x, Slice([0,hoffset,0],
                                    [-1,x_shape[1],-1])(concat_layer)])

        for _ in range(nconv):
            x = Conv1D(filters=nfilters, kernel_size=kernel_size, padding=padding)(x)
            receptive_field += (kernel_size-1) * 2**sum_of_strides
            hidden_layers.append(x)
            x = BatchNormalization()(x)
            x = ReLU()(x)
            if padding == 'valid':
                noutput_tics = noutput_tics - kernel_size + 1

        return x

    ds = []
    for i,_ in enumerate(range(ndown)):
        x = down_block_1D(x, nfilters=nfilters*(2**i))
        ds.append(x)
        if model_parameters['downsample']=='pool':
            x = MaxPooling1D(pool_size=2, strides=2)(x)
            noutput_tics = math.floor(noutput_tics / 2)
        else:
            x = Conv1D(filters=nfilters*(2**i), kernel_size=2, strides=2)(x)
            noutput_tics = math.ceil((noutput_tics - 2 + 1) / 2)
        receptive_field += (2-1) * 2**sum_of_strides
        sum_of_strides += 2 - 1
    x = down_block_1D(x, nfilters=nfilters*(2**ndown))

    if dropout>0:
        x = Dropout(dropout)(x)

    for i,_ in enumerate(range(nup)):
        x = up_block_1D(x, ds.pop(), nfilters=nfilters*(2**(ndown-i-1)))

    receptive_field *= stride_tics
    
    print("receptive_field_time = %d tics = %f %s" % (receptive_field,
        receptive_field / audio_tic_rate / time_scale, time_units), file=io)

    if model_parameters['output']=='conv':
        x = Conv1D(filters=model_settings['nlabels'], kernel_size=noutput_tics)(x)
    else:
        x = Slice([ 0, math.floor(noutput_tics/2), 0],
                  [-1, x.shape[1]-noutput_tics+1, -1])(x)
        x = Conv1D(filters=model_settings['nlabels'], kernel_size=1)(x)
    hidden_layers.append(x)

    final_layer = Reshape((-1,model_settings['nlabels']))(x)

    print('u-net.py version = 0.1', file=io)
    return tf.keras.Model(inputs=input_layer, outputs=[hidden_layers, final_layer],
                          name='u-net')
