import sys
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
    changed, window_sec2 = M.next_pow2_sec(float(V.model_parameters['window'].value) * M.time_scale)
    if changed:
        bokehlog.info("WARNING:  adjusting `window ("+M.time_units+")` to be a power of two in tics")
        V.model_parameters['window'].css_classes = ['changed']
        V.model_parameters['window'].value = str(window_sec2 / M.time_scale)
        ps.append('window')
    mel, _ = V.model_parameters['mel_dct'].value.split(',')
    nfreqs = round(window_sec2*M.audio_tic_rate/2+1)
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
    stride_sec = float(V.model_parameters['stride'].value) * M.time_scale
    stride_tics = round(M.audio_tic_rate * stride_sec)
    stride_sec2 = stride_tics / M.audio_tic_rate
    nconvlayers = int(V.model_parameters['nconvlayers'].value)
    nstride_time = len(parse_layers(V.model_parameters['stride_time'].value, nconvlayers))
    downsample_by = 2 ** nstride_time
    output_tic_rate = M.audio_tic_rate / stride_tics / downsample_by
    if output_tic_rate != round(output_tic_rate) or stride_sec2 != stride_sec:
        if output_tic_rate == round(output_tic_rate):
            bokehlog.info("WARNING:  adjusting `stride ("+M.time_units+")` to be an integer number of tics")
        else:
            bokehlog.info("WARNING:  adjusting `stride ("+M.time_units+")` such that the output sampling rate is an integer number")
            downsampled_rate = M.audio_tic_rate / downsample_by
            if downsampled_rate != round(downsampled_rate):
                stride_sec2 = "-1"
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
                    stride_sec2 = stride_tics2 / M.audio_tic_rate
                else:
                    stride_sec2 = "-1"
                    bokehlog.info("ERROR:  downsampling achieved by `stride after` is prime")
        V.model_parameters['stride'].css_classes = ['changed']
        V.model_parameters['stride'].value = str(stride_sec2 * M.time_scale)
        if V.bokeh_document:
            V.bokeh_document.add_next_tick_callback(lambda: _callback(['stride_sec'],M,V,C))
        else:
            _callback(['stride_sec'],M,V,C)

def mel_dct_callback(n,M,V,C):
    if V.model_parameters['mel_dct'].value.count(',') != 1:
        bokehlog.info("ERROR:  `mel & DCT` must to two positive integers separated by a comma.")
        return
    mel, dct = V.model_parameters['mel_dct'].value.split(',')
    if int(dct) > int(mel):
        bokehlog.info("WARNING:  adjusting `mel & DCT` such that DCT is less than or equal to mel")
        V.model_parameters['mel_dct'].css_classes = ['changed']
        V.model_parameters['mel_dct'].value = mel+','+mel
        if V.bokeh_document:
            V.bokeh_document.add_next_tick_callback(lambda: _callback(['mel_dct'],M,V,C))
        else:
            _callback(['mel_dct'],M,V,C)

def range_callback(n,M,V,C):
    if V.model_parameters['range'].value=="":  return
    if V.model_parameters['range'].value.count('-') != 1:
        bokehlog.info("ERROR:  `range ("+M.freq_units+")`, if not blank, must to two non-negative numbers separated by a hyphen.")
        return
    lo, hi = V.model_parameters['range'].value.split('-')
    lo = float(lo) * M.freq_scale
    hi = float(hi) * M.freq_scale
    if hi > M.audio_tic_rate // 2:  hi=M.audio_tic_rate // 2;
    if lo > hi:  lo=0;
    if V.model_parameters['range'].value != str(lo/M.freq_scale)+'-'+str(hi/M.freq_scale):
        bokehlog.info("WARNING:  adjusting `range ("+M.freq_units+")` such that lower bound is not negative and the higher bound less than the Nyquist frequency.")
        V.model_parameters['range'].css_classes = ['changed']
        V.model_parameters['range'].value = str(lo/M.freq_scale)+'-'+str(hi/M.freq_scale)
        if V.bokeh_document:
            V.bokeh_document.add_next_tick_callback(lambda: _callback(['range'],M,V,C))
        else:
            _callback(['range'],M,V,C)

def fused_callback(n,M,V,C):
    dilate_stride_callback("stride_time",n,M,V,C)
    stride_time_callback(n,M,V,C)

def dilate_stride_callback(key,n,M,V,C):
    nconvlayers = int(V.model_parameters['nconvlayers'].value)
    stride_time = parse_layers(V.model_parameters['stride_time'].value, nconvlayers)
    stride_freq = parse_layers(V.model_parameters['stride_freq'].value, nconvlayers)
    dilate_time = parse_layers(V.model_parameters['dilate_time'].value, nconvlayers)
    dilate_freq = parse_layers(V.model_parameters['dilate_freq'].value, nconvlayers)
    stride = set(stride_time + stride_freq)
    dilate = set(dilate_time + dilate_freq)
    if stride & dilate:
        bokehlog.info("WARNING:  adjusting `"+key+"` so that the convolutional layers with strides do not overlap with those that dilate")
        V.model_parameters[key].css_classes = ['changed']
        tmp = set(parse_layers(V.model_parameters[key].value, nconvlayers))
        V.model_parameters[key].value = esrap_layers(list(tmp - (stride & dilate)), nconvlayers)
        if V.bokeh_document:
            V.bokeh_document.add_next_tick_callback(lambda: _callback([key],M,V,C))
        else:
            _callback([key],M,V,C)

def stride_time_callback(n,M,V,C):
    nconvlayers = int(V.model_parameters['nconvlayers'].value)
    stride_time = parse_layers(V.model_parameters['stride_time'].value, nconvlayers)
    downsampled_rate = M.audio_tic_rate / 2 ** len(stride_time)
    if downsampled_rate != round(downsampled_rate):
        bokehlog.info("WARNING:  adjusting `stride time` such that the downsampled rate achieved in conjunction with `# layers` is an integer")
        while downsampled_rate != round(downsampled_rate):
            stride_time.pop()
            downsampled_rate = M.audio_tic_rate / 2 ** min(nconvlayers, len(stride_time))
        V.model_parameters['stride_time'].css_classes = ['changed']
        V.model_parameters['stride_time'].value = esrap_layers(stride_time, nconvlayers)
        if V.bokeh_document:
            V.bokeh_document.add_next_tick_callback(lambda: _callback(['stride_time'],M,V,C))
        else:
            _callback(['stride_time'],M,V,C)

def nlayers_callback(n,M,V,C):
    nconvlayers = int(V.model_parameters['nconvlayers'].value)
    stride_time = parse_layers(V.model_parameters['stride_time'].value, nconvlayers)
    downsampled_rate = M.audio_tic_rate / 2 ** len(stride_time)
    if downsampled_rate != round(downsampled_rate):
        bokehlog.info("WARNING:  adjusting `# conv layers` such that the downsampled rate achieved in conjunction with `stride time` is an integer")
        for this_nconvlayers in range(nconvlayers,0,-1):
            stride_time = [x for x in stride_time if x <= this_nconvlayers]
            downsampled_rate = M.audio_tic_rate / 2 ** min(this_nconvlayers, len(stride_time))
            if downsampled_rate == round(downsampled_rate):
                break
        V.model_parameters['nconvlayers'].css_classes = ['changed']
        V.model_parameters['nconvlayers'].value = str(this_nconvlayers)
        if V.bokeh_document:
            V.bokeh_document.add_next_tick_callback(lambda: _callback(['nconvlayers'],M,V,C))
        else:
            _callback(['nconvlayers'],M,V,C)

use_audio=1
use_video=0

def model_parameters(time_units, freq_units, time_scale, freq_scale):
    return [
        ["representation",  "representation",           ['waveform',
                                                         'spectrogram',
                                                         'mel-cepstrum'],     'mel-cepstrum',         1, [],                 None,                                                          True],
        ["window",          "window ("+time_units+")",  '',                   str(0.0064/time_scale), 1, ["representation",
                                                                                                          ["spectrogram",
                                                                                                           "mel-cepstrum"]], window_callback,                                               True],
        ["stride",          "stride ("+time_units+")",  '',                   str(0.0016/time_scale), 1, ["representation",
                                                                                                          ["spectrogram",
                                                                                                           "mel-cepstrum"]], stride_callback,                                               True],
        ["range",           "range ("+freq_units+")",   '',                   '',                     1, ["representation",
                                                                                                          ["spectrogram"]],  range_callback,                                                False],
        ["mel_dct",         "mel & DCT",                '',                   '7,7',                  1, ["representation",
                                                                                                          ["mel-cepstrum"]], mel_dct_callback,                                              True],
        ["connection_type", "connection",               ['plain',
                                                         'residual'],         'plain',                1, [],                 None,                                                          True],
        ["nconvlayers",     "# conv layers",            '',                   '2',                    1, [],                 nlayers_callback,                                              True],
        ["kernel_sizes",    "kernels",                  '',                   '5x5,3',                1, [],                 None,                                                          True],
        ["nfeatures",       "# features",               '',                   '64,64',                1, [],                 None,                                                          True],
        ["normalization",   "normalization",            ['none',
                                                         'batch before ReLU',
                                                         'batch after ReLU'], 'none',                 1, [],                 None,                                                          True],
        ["stride_time",     "stride time",              '',                   '',                     1, [],                 fused_callback,                                                False],
        ["stride_freq",     "stride freq",              '',                   '',                     1, ["representation",
                                                                                                          ["spectrogram",
                                                                                                           "mel-cepstrum"]], lambda n,M,V,C: dilate_stride_callback("stride_freq",n,M,V,C), False],
        ["dilate_time",     "dilate time",              '',                   '',                     1, [],                 lambda n,M,V,C: dilate_stride_callback("dilate_time",n,M,V,C), False],
        ["dilate_freq",     "dilate freq",              '',                   '',                     1, ["representation",
                                                                                                          ["spectrogram",
                                                                                                           "mel-cepstrum"]], lambda n,M,V,C: dilate_stride_callback("dilate_freq",n,M,V,C), False],
        ["dropout",         "dropout %",                '',                   '50',                   1, [],                 None,                                                          True],
        ["pool_kind",       "pool kind",                ["none",
                                                         "max",
                                                         "average"],          "none",                 1, [],                 None,                                                          True],
        ["pool_size",       "pool size",                '',                   '2,2',                  1, ["pool_kind",
                                                                                                          ["max",
                                                                                                           "average"]],      None,                                                          True],
        ["denselayers",     "dense layers",             '',                   '',                     1, [],                 None,                                                          False],
        ["augment_volume",  "augment volume",           '',                   '1,1',                  1, [],                 None,                                                          True],
        ["augment_noise",   "augment noise",            '',                   '0,0',                  1, [],                 None,                                                          True],
    ]

class Augment(tf.keras.layers.Layer):
    def __init__(self, volume_range, noise_range, **kwargs):
        super(Augment, self).__init__(**kwargs)
        self.volume_range = volume_range
        self.noise_range = noise_range
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'volume_range': self.volume_range,
            'noise_range': self.noise_range,
        })
        return config
    def call(self, inputs, training=None):
        if not training:
            return inputs
        if self.volume_range != [1,1] or self.noise_range != [0,0]:
            nbatch_1_nchannel = tf.stack((tf.shape(inputs)[0], 1, tf.shape(inputs)[2]), axis=0)
        if self.volume_range != [1,1]:
            volume_ranges = tf.random.uniform(nbatch_1_nchannel, *self.volume_range)
            inputs = tf.math.multiply(volume_ranges, inputs)
        if self.noise_range != [0,0]:
            noise_ranges = tf.random.uniform(nbatch_1_nchannel, *self.noise_range)
            noises = tf.random.normal(tf.shape(inputs), 0, noise_ranges)
            inputs = tf.math.add(noises, inputs)
        return inputs

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

def esrap_layers(arg, nconvlayers):
    if not arg: return ""
    arg_appended = arg + [arg[-1]+2]
    last_elt = start = arg_appended[0]
    stride_time = ""
    for elt in arg_appended[1:]:
        if last_elt+1 != elt:
            stop = last_elt
            if stride_time:  stride_time += ","
            if start == stop:
                stride_time += str(start)
            elif start==0:
                stride_time += "<=" + str(stop)
            elif stop==nconvlayers:
                stride_time += ">=" + str(start)
            else:
                stride_time += str(start) + "-" + str(stop)
            start = elt
        last_elt = elt
    return stride_time 

def parse_layers(arg, nconvlayers):
    arg = arg.split(',') if ',' in arg else [arg]
    layers = []
    for elt in arg:
        if '-' in elt:
            elts = elt.split('-')
            if len(elts) != 2:
                bokehlog.info("WARNING: invalid stride or dilation layers specification: ", elt, " in ", arg)
            layers.extend(range(int(elts[0]), int(elts[1])+1))
        elif elt.startswith('<='):
            layers.extend(range(1,int(elt[2:])+1))
        elif elt.startswith('>='):
            layers.extend(range(int(elt[2:]), nconvlayers+1))
        elif elt.startswith('<'):
            layers.extend(range(1,int(elt[1:])))
        elif elt.startswith('>'):
            layers.extend(range(int(elt[1:])+1, nconvlayers+1))
        elif elt:
            layers.append(int(elt))
    return sorted([x for x in layers if x<=nconvlayers])

def dilation(iconv, dilate_time, dilate_freq):
  return [2**(sum([x<=iconv for x in dilate_time])),
          2**(sum([x<=iconv for x in dilate_freq]))]

def create_model(model_settings, model_parameters, io=sys.stdout):
  audio_tic_rate = model_settings['audio_tic_rate']
  time_units = model_settings['time_units']
  freq_units = model_settings['freq_units']
  time_scale = model_settings['time_scale']
  freq_scale = model_settings['freq_scale']
  representation = model_parameters['representation']
  kernel_sizes = model_parameters['kernel_sizes'].split(',')
  kernel_sizes[0] = [int(x) for x in kernel_sizes[0].split('x')] \
                    if 'x' in kernel_sizes[0] else int(kernel_sizes[0])
  kernel_sizes[1] = int(kernel_sizes[1])
  nconvlayers = int(model_parameters['nconvlayers'])
  denselayers = [] if model_parameters['denselayers']=='' \
                   else [int(x) for x in model_parameters['denselayers'].split(',')]
  nfeatures = [int(x) for x in model_parameters['nfeatures'].split(',')]
  stride_time = parse_layers(model_parameters['stride_time'], nconvlayers)
  stride_freq = parse_layers(model_parameters['stride_freq'], nconvlayers)
  dilate_time = parse_layers(model_parameters['dilate_time'], nconvlayers)
  dilate_freq = parse_layers(model_parameters['dilate_freq'], nconvlayers)
  use_residual = model_parameters['connection_type']=='residual'
  dropout = float(model_parameters['dropout'])/100
  if model_parameters['pool_kind']=='max':
    pool_kind = MaxPool2D
  elif model_parameters['pool_kind']=='average':
    pool_kind = AveragePooling2D
  else:
    pool_kind = None
  if pool_kind:
    pool_size = [int(x) for x in model_parameters['pool_size'].split(',')]
  normalize_before = 'before' in model_parameters['normalization']
  normalize_after = 'after' in model_parameters['normalization']

  if representation == "waveform":
    window_tics = stride_tics = 1
  else:
    window_tics = round(audio_tic_rate * float(model_parameters['window']) * time_scale)
    stride_tics = round(audio_tic_rate * float(model_parameters['stride']) * time_scale)

    if not (window_tics & (window_tics-1) == 0) or window_tics == 0:
      next_higher_tic = np.power(2, np.ceil(np.log2(window_tics))).astype(int)
      next_lower_tic = np.power(2, np.floor(np.log2(window_tics))).astype(int)
      sigdigs = np.ceil(np.log10(next_higher_tic)).astype(int)+1
      next_higher_sec = np.around(next_higher_tic/audio_tic_rate, decimals=sigdigs)
      next_lower_sec = np.around(next_lower_tic/audio_tic_rate, decimals=sigdigs)
      raise Exception("ERROR: 'window ("+time_units+")' should be a power of two when converted to tics.  "+
                      model_parameters['window']+" "+time_units+" is "+str(window_tics)+" tics for Fs="+
                      str(audio_tic_rate)+".  try "+str(next_lower_sec)+" "+time_units+" (="+str(next_lower_tic)+
                      ") or "+str(next_higher_sec)+" "+time_units+" (="+str(next_higher_tic)+") instead.")

  downsample_by = 2 ** len(stride_time)
  output_tic_rate = audio_tic_rate / stride_tics / downsample_by
  print('downsample_by = '+str(downsample_by), file=io)
  print('output_tic_rate = '+str(output_tic_rate), file=io)
  if output_tic_rate != round(output_tic_rate):
    raise Exception("ERROR: "+str(1/time_scale)+" / 'stride ("+time_units+")' should be an integer multiple of the downsampling rate achieved by `stride after`")

  hidden_layers = []

  context_tics = int(model_settings['audio_tic_rate'] * model_settings['context'] * time_scale)
  ninput_tics = context_tics + (model_settings['parallelize']-1) * stride_tics * downsample_by
  noutput_tics = (context_tics-window_tics) // stride_tics + 1

  inputs = Input(shape=(ninput_tics, model_settings['audio_nchannels']))
  hidden_layers.append(inputs)
  
  volume_range = [float(x) for x in model_parameters['augment_volume'].split(',')]
  noise_range = [float(x) for x in model_parameters['augment_noise'].split(',')]
  if volume_range != [1,1] or noise_range != [0,0]:
    x = Augment(volume_range, noise_range)(inputs)
  else:
    x = inputs

  if representation == "waveform":
    x = Reshape((ninput_tics,1,model_settings['audio_nchannels']))(x)
  elif representation == "spectrogram":
    x = Spectrogram(window_tics, stride_tics)(x)
    if model_parameters['range'] != "":
      lo, hi = model_parameters['range'].split('-')
      lo = float(lo) * freq_scale
      hi = float(hi) * freq_scale
      nyquist = audio_tic_rate/2
      nfreqs = x.shape[2]
      x = Slice([0, 0, round(nfreqs * lo / nyquist), 0],
                [-1, -1, round(nfreqs * (hi - lo) / nyquist), -1])(x)
    hidden_layers.append(x)
  elif representation == "mel-cepstrum":
    filterbank_nchannels, dct_ncoefficients = model_parameters['mel_dct'].split(',')
    x = MelCepstrum(window_tics, stride_tics, audio_tic_rate,
                         int(filterbank_nchannels), int(dct_ncoefficients))(x)
    hidden_layers.append(x)
  x_shape = x.shape

  receptive_field = [1,1]
  sum_of_strides = [0,0]
  iconv=0
  dilation_rate = dilation(iconv+1, dilate_time, dilate_freq)

  # 2D convolutions
  dilated_kernel_size = [(kernel_sizes[0][0] - 1) * dilation_rate[0] + 1,
                         (kernel_sizes[0][1] - 1) * dilation_rate[1] + 1]
  while x_shape[1] >= dilated_kernel_size[0] and \
        x_shape[2] >= dilated_kernel_size[1] and \
        noutput_tics >= dilated_kernel_size[0] and \
        iconv<nconvlayers:
    if use_residual and iconv%2!=0:
      bypass = x
    strides=[1+(iconv+1 in stride_time), 1+(iconv+1 in stride_freq)]
    conv = Conv2D(nfeatures[0], kernel_sizes[0],
                  strides=strides, dilation_rate=dilation_rate)(x)
    receptive_field[0] += (dilated_kernel_size[0] - 1) * 2**sum_of_strides[0]
    receptive_field[1] += (dilated_kernel_size[1] - 1) * 2**sum_of_strides[1]
    sum_of_strides[0] += strides[0] - 1
    sum_of_strides[1] += strides[1] - 1
    if use_residual and iconv%2==0 and iconv>1:
      bypass_shape = bypass.shape
      conv_shape = conv.shape
      if bypass_shape[3]==conv_shape[3]:
        hoffset = (bypass_shape[1] - conv_shape[1]) // 2
        woffset = (bypass_shape[2] - conv_shape[2]) // 2
        conv = Add()([conv, Slice([0,hoffset,woffset,0],
                                  [-1,conv_shape[1],conv_shape[2],-1])(bypass)])
    hidden_layers.append(conv)
    if normalize_before:
      conv = BatchNormalization()(conv)
    x = ReLU()(conv)
    if normalize_after:
      x = BatchNormalization()(x)
    x_shape = x.shape
    noutput_tics = math.ceil((noutput_tics - dilated_kernel_size[0] + 1) / strides[0])
    iconv += 1
    dilation_rate = dilation(iconv+1, dilate_time, dilate_freq)
    dilated_kernel_size = [(kernel_sizes[0][0] - 1) * dilation_rate[0] + 1,
                           (kernel_sizes[0][1] - 1) * dilation_rate[1] + 1]

  # 1D convolutions (or actually, pan-freq 2D)
  dilated_kernel_size = (kernel_sizes[1] - 1) * dilation_rate[0] + 1
  while noutput_tics >= dilated_kernel_size and \
        iconv<nconvlayers:
    if use_residual and iconv%2!=0:
      bypass = x
    strides=[1+(iconv+1 in stride_time), 1]
    conv = Conv2D(nfeatures[1], (kernel_sizes[1], x_shape[2]),
                  strides=strides, dilation_rate=[dilation_rate[0], 1])(x)
    receptive_field[0] += (dilated_kernel_size - 1) * 2**sum_of_strides[0]
    sum_of_strides[0] += strides[0] - 1
    if use_residual and iconv%2==0 and iconv>1:
      bypass_shape = bypass.shape
      conv_shape = conv.shape
      if bypass_shape[2]==conv_shape[2] and bypass_shape[3]==conv_shape[3]:
        offset = (bypass_shape[1] - conv_shape[1]) // 2
        conv = Add()([conv, Slice([0,offset,0,0],[-1,conv_shape[1],-1,-1])(bypass)])
    hidden_layers.append(conv)
    if normalize_before:
      conv = BatchNormalization()(conv)
    x = ReLU()(conv)
    if normalize_after:
      x = BatchNormalization()(x)
    x_shape = x.shape
    noutput_tics = math.ceil((noutput_tics - dilated_kernel_size + 1) / strides[0])
    iconv += 1
    dilation_rate = dilation(iconv+1, dilate_time, dilate_freq)
    dilated_kernel_size = (kernel_sizes[1] - 1) * dilation_rate[0] + 1

  receptive_field[0] *= stride_tics
  
  print("receptive_field_time = %d tics = %f %s" % (receptive_field[0],
      receptive_field[0] / audio_tic_rate / time_scale, time_units), file=io)
  print("receptive_field_freq = %d bins = %f %s" % (receptive_field[1],
      receptive_field[1] * audio_tic_rate / window_tics / freq_scale, freq_units), file=io)

  if dropout>0:
      x = Dropout(dropout)(x)

  if pool_kind:
    x = pool_kind(pool_size=pool_size, strides=pool_size)(x)
    x_shape = x.shape
    noutput_tics = math.floor(noutput_tics / pool_size[0])

  # final dense layers (or actually, pan-freq pan-time 2D convs)
  for idense, nunits in enumerate(denselayers+[model_settings['nlabels']]):
    if idense>0:
      x = ReLU()(x)
      if dropout>0:
          x = Dropout(dropout)(x)
    x = Conv2D(nunits, (noutput_tics if idense==0 else 1, x_shape[2]))(x)
    hidden_layers.append(conv)
    x_shape = x.shape

  final = Reshape((-1,model_settings['nlabels']))(x)

  print('convolutional.py version = 0.1.1', file=io)
  return tf.keras.Model(inputs=inputs, outputs=[hidden_layers, final], name="convolutional")
