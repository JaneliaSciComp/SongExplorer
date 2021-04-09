import math

import tensorflow as tf
from tensorflow.keras.layers import *

model_parameters = [
  # key in `model_settings`, title in GUI, '' for textbox or [] for pull-down, default value
  ["kernel_sizes",       "kernels",          '', '5,3'],
  ["nlayers",            "# layers",         '', '2'],
  ["nfeatures",          "# features",       '', '64,64,64'],
  ["dilate_after_layer", "dilate after",     '', '65535'],
  ["stride_after_layer", "stride after",     '', '65535'],
  ["connection_type",    "connection",       ["plain", "residual"], 'plain'],
  ["dropout",            "dropout",          '', '0.5'],
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
  dropout_prob = float(model_settings['dropout'])
  input_frequency_size = model_settings['input_frequency_size']
  input_time_size = model_settings['input_time_size']
  input_channel_size = model_settings['channel_count']
  filter_counts = [int(x) for x in model_settings['nfeatures'].split(',')]
  filter_sizes = [int(x) for x in model_settings['kernel_sizes'].split(',')]
  nlayers = int(model_settings['nlayers'])
  nwindows = model_settings['nwindows']
  dilate_after_layer = int(model_settings['dilate_after_layer'])
  stride_after_layer = int(model_settings['stride_after_layer'])
  residual = True if model_settings['connection_type']=='residual' else False
  label_count = model_settings['label_count']

  iconv=0
  hidden_layers = []

  inputs0 = Input(shape=(input_channel_size * input_time_size * input_frequency_size))
  reshaped = Reshape((input_channel_size, input_time_size, input_frequency_size))(inputs0)
  permuted = Permute((2,3,1))(reshaped)
  hidden_layers.append(permuted)
  inputs_shape = permuted.get_shape().as_list()
  output_time_size1 = inputs_shape[1]-nwindows+1

  # 2D convolutions
  inputs = permuted
  while inputs_shape[2]>=filter_sizes[0] and iconv<nlayers:
    if residual and iconv%2==0:
      bypass = inputs
    strides=[1+(iconv>=stride_after_layer), 1]
    dilation_rate=[2**max(0,iconv-dilate_after_layer+1), 1]
    conv = Conv2D(filter_counts[0], filter_sizes[0],
                  strides=strides, dilation_rate=dilation_rate)(inputs)
    if residual and iconv%2!=0:
      bypass_shape = bypass.get_shape().as_list()
      output_shape = conv.get_shape().as_list()
      woffset = (bypass_shape[1] - output_shape[1]) // 2
      hoffset = (bypass_shape[2] - output_shape[2]) // 2
      conv = Add()([conv, Slice([0,hoffset,woffset,0],
                                [-1,output_shape[1],output_shape[2],-1])(bypass)])
    hidden_layers.append(conv)
    relu = ReLU()(conv)
    inputs = Dropout(dropout_prob)(relu)
    inputs_shape = inputs.get_shape().as_list()
    output_time_size1 = math.ceil((output_time_size1 - filter_sizes[0] + 1) / strides[0])
    iconv += 1

  # 1D convolutions (or actually, pan-freq 2D)
  while inputs_shape[1]>=filter_sizes[1] and iconv<nlayers:
    if residual and iconv%2==0:
      bypass = inputs
    strides=[1+(iconv>=stride_after_layer), 1]
    dilation_rate=[2**max(0,iconv-dilate_after_layer+1), 1]
    conv = Conv2D(filter_counts[1], (filter_sizes[1], inputs_shape[2]),
                  strides=strides,
                  dilation_rate=dilation_rate)(inputs)
    if residual and iconv%2!=0:
      output_time_size = conv.get_shape().as_list()[1]
      offset = (bypass.get_shape().as_list()[1] - output_time_size) // 2
      conv = Add()([conv, Slice([0,offset,0,0],[-1,output_time_size,-1,-1])(bypass)])
    hidden_layers.append(conv)
    relu = ReLU()(conv)
    inputs = Dropout(dropout_prob)(relu)
    inputs_shape = inputs.get_shape().as_list()
    output_time_size1 = math.ceil((output_time_size1 - filter_sizes[1] + 1) / strides[0])
    iconv += 1

  # a final dense layer (or actually, pan-freq pan-time 2D conv)
  strides=1+(iconv>=stride_after_layer)
  final = Conv2D(label_count, (output_time_size1, inputs_shape[2]), strides=strides)(inputs)
  final = Reshape((-1,model_settings['nlabels']))(final)

  return tf.keras.Model(inputs=inputs0, outputs=[hidden_layers, final])
