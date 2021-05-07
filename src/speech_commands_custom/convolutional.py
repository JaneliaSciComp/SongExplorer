import math

import tensorflow as tf
from tensorflow.keras.layers import *

model_parameters = [
  # key in `model_settings`, title in GUI, '' for textbox or [] for pull-down, default value
  ["representation",     "representation", ['waveform',
                                            'spectrogram',
                                            'mel-cepstrum'], 'mel-cepstrum'],
  ["window_ms",          "window (msec)",  '',               '6.4'],
  ["stride_ms",          "stride (msec)",  '',               '1.6'],
  ["mel_dct",            "mel & DCT",      '',               '7,7'],
  ["kernel_sizes",       "kernels",        '',               '5,3'],
  ["nlayers",            "# layers",       '',               '2'],
  ["nfeatures",          "# features",     '',               '64,64'],
  ["dilate_after_layer", "dilate after",   '',               '65535'],
  ["stride_after_layer", "stride after",   '',               '65535'],
  ["connection_type",    "connection",     ['plain',
                                            'residual'],     'plain'],
  ["dropout",            "dropout",        '',               '0.5'],
  ]

class Spectrogram(tf.keras.layers.Layer):
    def __init__(self, window_tics, stride_tics, **kwargs):
        super(Spectrogram, self).__init__(**kwargs)
        self.window_tics = window_tics
        self.stride_tics = stride_tics
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'window_tics': self.window_tics,
            'stride_tics': self.stride_tics,
        })
        return config
    def call(self, inputs):
        # given channel X time, tf.signal.stft returns channel X time X freq
        inputs_bct = tf.transpose(inputs, [0,2,1])
        spectrograms = tf.math.abs(tf.signal.stft(inputs_bct,
                                                  self.window_tics, self.stride_tics))
        return tf.transpose(spectrograms, [0,2,3,1])

class MelCepstrum(tf.keras.layers.Layer):
    def __init__(self, window_tics, stride_tics, audio_tic_rate,
                       filterbank_nchannels, dct_ncoefficients, **kwargs):
        super(MelCepstrum, self).__init__(**kwargs)
        self.window_tics = window_tics
        self.stride_tics = stride_tics
        self.audio_tic_rate = audio_tic_rate
        self.filterbank_nchannels = filterbank_nchannels
        self.dct_ncoefficients = dct_ncoefficients
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'window_tics': self.window_tics,
            'stride_tics': self.stride_tics,
            'audio_tic_rate ': self.audio_tic_rate ,
            'filterbank_nchannels': self.filterbank_nchannels,
            'dct_ncoefficients': self.dct_ncoefficients,
        })
        return config
    def call(self, inputs):
        inputs_bct = tf.transpose(inputs, [0,2,1])
        spectrograms = tf.math.abs(tf.signal.stft(inputs_bct,
                                                  self.window_tics, self.stride_tics))

        # Warp the linear scale spectrograms into the mel-scale.
        num_spectrogram_bins = self.window_tics//2+1
        lower_edge_hertz = 0.0
        upper_edge_hertz = self.audio_tic_rate//2
        num_mel_bins = self.filterbank_nchannels
        linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
          num_mel_bins, num_spectrogram_bins, self.audio_tic_rate,
          lower_edge_hertz, upper_edge_hertz)
        mel_spectrograms = tf.tensordot(
          spectrograms, linear_to_mel_weight_matrix, 1)
        mel_spectrograms.set_shape(spectrograms.shape[:-1].concatenate(
          linear_to_mel_weight_matrix.shape[-1:]))

        # Compute a stabilized log to get log-magnitude mel-scale spectrograms.
        log_mel_spectrograms = tf.math.log(mel_spectrograms + 1e-6)

        # Compute MFCCs from log_mel_spectrograms and take the first few
        mfccs = tf.signal.mfccs_from_log_mel_spectrograms(
          log_mel_spectrograms)[..., :self.dct_ncoefficients]

        return tf.transpose(mfccs, [0,2,3,1])

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
  audio_tic_rate = model_settings['audio_tic_rate']
  representation = model_settings['representation']
  if representation == "waveform":
    window_tics = stride_tics = 1
  else:
    window_tics = int(audio_tic_rate * float(model_settings['window_ms']) / 1000)
    stride_tics = int(audio_tic_rate * float(model_settings['stride_ms']) / 1000)
  kernel_sizes = [int(x) for x in model_settings['kernel_sizes'].split(',')]
  nlayers = int(model_settings['nlayers'])
  nfeatures = [int(x) for x in model_settings['nfeatures'].split(',')]
  dilate_after_layer = int(model_settings['dilate_after_layer'])
  stride_after_layer = int(model_settings['stride_after_layer'])
  use_residual = True if model_settings['connection_type']=='residual' else False
  dropout = float(model_settings['dropout'])

  iconv=0
  hidden_layers = []

  downsample_by = max(0, nlayers - stride_after_layer + 1)
  ninput_tics = model_settings['context_tics'] + \
           (model_settings['parallelize']-1) * stride_tics * 2 ** downsample_by
  noutput_tics = (model_settings['context_tics']-window_tics) // stride_tics + 1

  inputs0 = Input(shape=(ninput_tics, model_settings['nchannels']))

  if representation == "waveform":
    inputs = Reshape((ninput_tics,1,model_settings['nchannels']))(inputs0)
  elif representation == "spectrogram":
    inputs = Spectrogram(window_tics, stride_tics)(inputs0)
  elif representation == "mel-cepstrum":
    filterbank_nchannels, dct_ncoefficients = model_settings['mel_dct'].split(',')
    inputs = MelCepstrum(window_tics, stride_tics, audio_tic_rate,
                         int(filterbank_nchannels), int(dct_ncoefficients))(inputs0)
  hidden_layers.append(inputs)
  inputs_shape = inputs.get_shape().as_list()

  # 2D convolutions
  while inputs_shape[2]>=kernel_sizes[0] and iconv<nlayers:
    if use_residual and iconv%2!=0:
      bypass = inputs
    strides=[1+(iconv>=stride_after_layer), 1]
    dilation_rate=[2**max(0,iconv-dilate_after_layer+1), 1]
    conv = Conv2D(nfeatures[0], kernel_sizes[0],
                  strides=strides, dilation_rate=dilation_rate)(inputs)
    if use_residual and iconv%2==0 and iconv>1:
      bypass_shape = bypass.get_shape().as_list()
      conv_shape = conv.get_shape().as_list()
      if bypass_shape[3]==conv_shape[3]:
        hoffset = (bypass_shape[1] - conv_shape[1]) // 2
        woffset = (bypass_shape[2] - conv_shape[2]) // 2
        conv = Add()([conv, Slice([0,hoffset,woffset,0],
                                  [-1,conv_shape[1],conv_shape[2],-1])(bypass)])
    hidden_layers.append(conv)
    relu = ReLU()(conv)
    inputs = Dropout(dropout)(relu)
    inputs_shape = inputs.get_shape().as_list()
    noutput_tics = math.ceil((noutput_tics - kernel_sizes[0] + 1) / strides[0])
    iconv += 1

  # 1D convolutions (or actually, pan-freq 2D)
  while inputs_shape[1]>=kernel_sizes[1] and iconv<nlayers:
    if use_residual and iconv%2!=0:
      bypass = inputs
    strides=[1+(iconv>=stride_after_layer), 1]
    dilation_rate=[2**max(0,iconv-dilate_after_layer+1), 1]
    conv = Conv2D(nfeatures[1], (kernel_sizes[1], inputs_shape[2]),
                  strides=strides,
                  dilation_rate=dilation_rate)(inputs)
    if use_residual and iconv%2==0 and iconv>1:
      bypass_shape = bypass.get_shape().as_list()
      conv_shape = conv.get_shape().as_list()
      if bypass_shape[3]==conv_shape[3]:
        offset = (bypass_shape[1] - conv_shape[1]) // 2
        conv = Add()([conv, Slice([0,offset,0,0],[-1,conv_shape[1],-1,-1])(bypass)])
    hidden_layers.append(conv)
    relu = ReLU()(conv)
    inputs = Dropout(dropout)(relu)
    inputs_shape = inputs.get_shape().as_list()
    noutput_tics = math.ceil((noutput_tics - kernel_sizes[1] + 1) / strides[0])
    iconv += 1

  # a final dense layer (or actually, pan-freq pan-time 2D conv)
  strides=1+(iconv>=stride_after_layer)
  final = Conv2D(model_settings['nlabels'], (noutput_tics, inputs_shape[2]),
                 strides=strides)(inputs)
  final = Reshape((-1,model_settings['nlabels']))(final)

  return tf.keras.Model(inputs=inputs0, outputs=[hidden_layers, final])
