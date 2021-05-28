import math

import tensorflow as tf
from tensorflow.keras.layers import *

import numpy as np
from itertools import chain, zip_longest

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
        bokehlog.info("WARNING:  adjusting `window (msec)` to be a power of two in tics")
        V.model_parameters['window_ms'].css_classes = ['changed']
        V.model_parameters['window_ms'].value = str(window_ms2)
        ps.append('window_ms')
    mel, _ = V.model_parameters['mel_dct'].value.split(',')
    nfreqs = round(window_ms2/1000*M.audio_tic_rate/2+1)
    if int(mel) != nfreqs:
        changed=True
        bokehlog.info("WARNING:  adjusting `mel & DCT` to both be equal to the number of frequencies")
        V.model_parameters['mel_dct'].css_classes = ['changed']
        V.model_parameters['mel_dct'].value = str(nfreqs)+','+str(nfreqs)
        ps.append('mel_dct')
    if changed:
        if V.bokeh_document:
            V.bokeh_document.add_next_tick_callback(lambda: _callback(ps,M,V,C))
        else:
            _callback(ps,M,V,C)

def stride_callback(n,M,V,C):
    stride_ms = float(V.model_parameters['stride_ms'].value)
    stride_tics = round(M.audio_tic_rate * stride_ms / 1000)
    stride_ms2 = stride_tics / M.audio_tic_rate * 1000
    downsample_by = 2 ** max(0, int(V.model_parameters['nlayers'].value) -
                              int(V.model_parameters['stride_after_layer'].value) + 1)
    output_tic_rate = M.audio_tic_rate / stride_tics / downsample_by
    if output_tic_rate != round(output_tic_rate) or stride_ms2 != stride_ms:
        if output_tic_rate == round(output_tic_rate):
            bokehlog.info("WARNING:  adjusting `stride (msec)` to be an integer number of tics")
        else:
            bokehlog.info("WARNING:  adjusting `stride (msec)` such that the output sampling rate is an integer number")
            downsampled_rate = M.audio_tic_rate / downsample_by
            if downsampled_rate != round(downsampled_rate):
                stride_ms2 = "-1"
                bokehlog.info("ERROR:  downsampling achieved by `stride after` results in non-integer sampling rate")
            else:
                for this_output_tic_rate in [x for x in chain.from_iterable(zip_longest(
                                             range(math.floor(output_tic_rate),0,-1),
                                             range(math.ceil(output_tic_rate),
                                                   math.floor(math.sqrt(downsampled_rate)))))
                                             if x is not None]:
                    stride_tics2 = downsampled_rate / this_output_tic_rate
                    if stride_tics2 == round(stride_tics2):
                        break
                if stride_tics2 == round(stride_tics2):
                    stride_ms2 = stride_tics2 / M.audio_tic_rate * 1000
                else:
                    stride_ms2 = "-1"
                    bokehlog.info("ERROR:  downsampling achieved by `stride after` is prime")
        V.model_parameters['stride_ms'].css_classes = ['changed']
        V.model_parameters['stride_ms'].value = str(stride_ms2)
        if V.bokeh_document:
            V.bokeh_document.add_next_tick_callback(lambda: _callback(['stride_ms'],M,V,C))
        else:
            _callback(['stride_ms'],M,V,C)

def mel_dct_callback(n,M,V,C):
    mel, dct = V.model_parameters['mel_dct'].value.split(',')
    if int(dct) > int(mel):
        bokehlog.info("WARNING:  adjusting `mel & DCT` such that DCT is less than or equal to mel")
        V.model_parameters['mel_dct'].css_classes = ['changed']
        V.model_parameters['mel_dct'].value = mel+','+mel
        if V.bokeh_document:
            V.bokeh_document.add_next_tick_callback(lambda: _callback(['mel_dct'],M,V,C))
        else:
            _callback(['mel_dct'],M,V,C)

def stride_after_layer_callback(n,M,V,C):
    nlayers = int(V.model_parameters['nlayers'].value)
    stride_after = int(V.model_parameters['stride_after_layer'].value)
    downsampled_rate = M.audio_tic_rate / 2 ** max(0, nlayers - stride_after + 1)
    if stride_after<0 or downsampled_rate != round(downsampled_rate):
        bokehlog.info("WARNING:  adjusting `stride after` such that the downsampled rate achieved in conjunction with `# layers` is an integer")
        for this_stride_after in range(stride_after,nlayers):
            downsampled_rate = M.audio_tic_rate / 2 ** max(0, nlayers - this_stride_after + 1)
            if downsampled_rate == round(downsampled_rate):
                break
        V.model_parameters['stride_after_layer'].css_classes = ['changed']
        V.model_parameters['stride_after_layer'].value = str(this_stride_after)
        if V.bokeh_document:
            V.bokeh_document.add_next_tick_callback(lambda: _callback(['stride_after_layer'],M,V,C))
        else:
            _callback(['stride_after_layer'],M,V,C)

def nlayers_callback(n,M,V,C):
    nlayers = int(V.model_parameters['nlayers'].value)
    stride_after = int(V.model_parameters['stride_after_layer'].value)
    downsampled_rate = M.audio_tic_rate / 2 ** max(0, nlayers - stride_after + 1)
    if nlayers<0 or downsampled_rate != round(downsampled_rate):
        bokehlog.info("WARNING:  adjusting `# layers` such that the downsampled rate achieved in conjunction with `stride after` is an integer")
        for this_nlayers in range(nlayers,0,-1):
            downsampled_rate = M.audio_tic_rate / 2 ** max(0, this_nlayers - stride_after + 1)
            if downsampled_rate == round(downsampled_rate):
                break
        V.model_parameters['nlayers'].css_classes = ['changed']
        V.model_parameters['nlayers'].value = str(this_nlayers)
        if V.bokeh_document:
            V.bokeh_document.add_next_tick_callback(lambda: _callback(['nlayers'],M,V,C))
        else:
            _callback(['nlayers'],M,V,C)

model_parameters = [
  # key in `model_settings`, title in GUI, '' for textbox or [] for pull-down, default value, enable logic, callback
  ["representation",     "representation", ['waveform',
                                            'spectrogram',
                                            'mel-cepstrum'], 'mel-cepstrum', [],                 None],
  ["window_ms",          "window (msec)",  '',               '6.4',          ["representation",
                                                                              ["spectrogram",
                                                                               "mel-cepstrum"]], window_callback],
  ["stride_ms",          "stride (msec)",  '',               '1.6',          ["representation",
                                                                              ["spectrogram",
                                                                               "mel-cepstrum"]], stride_callback],
  ["mel_dct",            "mel & DCT",      '',               '7,7',          ["representation",
                                                                              ["mel-cepstrum"]], mel_dct_callback],
  ["kernel_sizes",       "kernels",        '',               '5,3',          [],                  None],
  ["nlayers",            "# layers",       '',               '2',            [],                  nlayers_callback],
  ["nfeatures",          "# features",     '',               '64,64',        [],                  None],
  ["dilate_after_layer", "dilate after",   '',               '65535',        [],                  None],
  ["stride_after_layer", "stride after",   '',               '65535',        [],                  stride_after_layer_callback],
  ["connection_type",    "connection",     ['plain',
                                            'residual'],     'plain',        [],                  None],
  ["dropout",            "dropout",        '',               '0.5',          [],                  None],
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
  kernel_sizes = [int(x) for x in model_settings['kernel_sizes'].split(',')]
  nlayers = int(model_settings['nlayers'])
  nfeatures = [int(x) for x in model_settings['nfeatures'].split(',')]
  dilate_after_layer = int(model_settings['dilate_after_layer'])
  stride_after_layer = int(model_settings['stride_after_layer'])
  use_residual = model_settings['connection_type']=='residual'
  dropout = float(model_settings['dropout'])

  if representation == "waveform":
    window_tics = stride_tics = 1
  else:
    window_tics = round(audio_tic_rate * float(model_settings['window_ms']) / 1000)
    stride_tics = round(audio_tic_rate * float(model_settings['stride_ms']) / 1000)

    if not (window_tics & (window_tics-1) == 0) or window_tics == 0:
      next_higher = np.power(2, np.ceil(np.log2(window_tics))).astype(np.int)
      next_lower = np.power(2, np.floor(np.log2(window_tics))).astype(np.int)
      sigdigs = np.ceil(np.log10(next_higher)).astype(np.int)+1
      next_higher_ms = np.around(next_higher/audio_tic_rate*1000, decimals=sigdigs)
      next_lower_ms = np.around(next_lower/audio_tic_rate*1000, decimals=sigdigs)
      raise Exception("ERROR: 'window (msec)' should be a power of two when converted to tics.  "+
                      model_settings['window_ms']+" ms is "+str(window_tics)+" tics for Fs="+
                      str(audio_tic_rate)+".  try "+str(next_lower_ms)+" ms (="+str(next_lower)+
                      ") or "+str(next_higher_ms)+"ms (="+str(next_higher)+") instead.")

  downsample_by = 2 ** max(0, nlayers - stride_after_layer + 1)
  output_tic_rate = audio_tic_rate / stride_tics / downsample_by
  print('downsample_by = '+str(downsample_by))
  print('output_tic_rate = '+str(output_tic_rate))
  if output_tic_rate != round(output_tic_rate):
    raise Exception("ERROR: 1000 / 'stride (msec)' should be an integer multiple of the downsampling rate achieved by `stride after`")

  iconv=0
  hidden_layers = []

  ninput_tics = model_settings['context_tics'] + \
           (model_settings['parallelize']-1) * stride_tics * downsample_by
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

  receptive_field_tics = last_stride = 1

  # 2D convolutions
  while inputs_shape[2]>=kernel_sizes[0] and iconv<nlayers:
    if use_residual and iconv%2!=0:
      bypass = inputs
    strides=[1+(iconv>=stride_after_layer), 1]
    dilation_rate=[2**max(0,iconv-dilate_after_layer+1), 1]
    conv = Conv2D(nfeatures[0], kernel_sizes[0],
                  strides=strides, dilation_rate=dilation_rate)(inputs)
    dilated_kernel_size = (kernel_sizes[0] - 1) * dilation_rate[0] + 1
    receptive_field_tics += (dilated_kernel_size - 1) * last_stride
    last_stride = strides[0]
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
    inputs = SpatialDropout2D(dropout)(relu)
    inputs_shape = inputs.get_shape().as_list()
    noutput_tics = math.ceil((noutput_tics - dilated_kernel_size + 1) / strides[0])
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
    dilated_kernel_size = (kernel_sizes[1] - 1) * dilation_rate[0] + 1
    receptive_field_tics += (dilated_kernel_size - 1) * last_stride
    last_stride = strides[0]
    if use_residual and iconv%2==0 and iconv>1:
      bypass_shape = bypass.get_shape().as_list()
      conv_shape = conv.get_shape().as_list()
      if bypass_shape[3]==conv_shape[3]:
        offset = (bypass_shape[1] - conv_shape[1]) // 2
        conv = Add()([conv, Slice([0,offset,0,0],[-1,conv_shape[1],-1,-1])(bypass)])
    hidden_layers.append(conv)
    relu = ReLU()(conv)
    inputs = SpatialDropout2D(dropout)(relu)
    inputs_shape = inputs.get_shape().as_list()
    noutput_tics = math.ceil((noutput_tics - dilated_kernel_size + 1) / strides[0])
    iconv += 1

  receptive_field_tics *= stride_tics 
  print("receptive_field_tics = %d" % receptive_field_tics)

  # a final dense layer (or actually, pan-freq pan-time 2D conv)
  strides=1+(iconv>=stride_after_layer)
  final = Conv2D(model_settings['nlabels'], (noutput_tics, inputs_shape[2]),
                 strides=strides)(inputs)
  final = Reshape((-1,model_settings['nlabels']))(final)

  return tf.keras.Model(inputs=inputs0, outputs=[hidden_layers, final])
