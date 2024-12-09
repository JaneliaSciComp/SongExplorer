import sys
import math

# all imported packages must be in the container
import tensorflow as tf
from tensorflow.keras.layers import *

# use bokehlog.info() to print debugging messages
import logging 
bokehlog = logging.getLogger("songexplorer") 

def _callback(ps,M,V,C):
    C.time.sleep(0.5)
    for p in ps:
        V.model_parameters[p].stylesheets = [""]
    M.save_state_callback()
    V.buttons_update()

def window_callback(n,M,V,C):
    ps = []
    changed, window_sec2 = M.next_pow2_sec(float(V.model_parameters['window'].value) * M.time_scale)
    if changed:
        bokehlog.info("WARNING:  adjusting `window ("+M.time_units+")` to be a power of two in tics")
        V.model_parameters['window'].stylesheets = M.changed_style
        V.model_parameters['window'].value = str(window_sec2 / M.time_scale)
        ps.append('window')
    mel, _ = V.model_parameters['mel_dct'].value.split(',')
    nfreqs = round(window_sec2*M.audio_tic_rate/2+1)
    if int(mel) != nfreqs:
        changed=True
        bokehlog.info("WARNING:  adjusting `mel & DCT` to both be equal to the number of frequencies")
        V.model_parameters['mel_dct'].stylesheets = M.changed_style
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
        V.model_parameters['stride'].stylesheets = M.changed_style
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
        V.model_parameters['mel_dct'].stylesheets = M.changed_style
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
        V.model_parameters['range'].stylesheets = M.changed_style
        V.model_parameters['range'].value = str(lo/M.freq_scale)+'-'+str(hi/M.freq_scale)
        if V.bokeh_document:
            V.bokeh_document.add_next_tick_callback(lambda: _callback(['range'],M,V,C))
        else:
            _callback(['range'],M,V,C)

use_audio=1
use_video=0

# a list of lists specifying the architecture-specific hyperparameters in the GUI
def model_parameters(time_units, freq_units, time_scale, freq_scale):
    return [
          # [key, title in GUI, "" for textbox or [] for pull-down, default value, width, enable logic, callback, required]
          ["representation",  "representation",          ['waveform',
                                                          'spectrogram',
                                                          'mel-cepstrum'], 'mel-cepstrum',         1, [],                 None,             True],
          ["window",          "window ("+time_units+")", '',               str(0.0064/time_scale), 1, ["representation",
                                                                                                       ["spectrogram",
                                                                                                        "mel-cepstrum"]], window_callback,  True],
          ["stride",          "stride ("+time_units+")", '',               str(0.0016/time_scale), 1, ["representation",
                                                                                                       ["spectrogram",
                                                                                                        "mel-cepstrum"]], stride_callback,  True],
          ["range",           "range ("+freq_units+")",  '',               '',                     1, ["representation",
                                                                                                       ["spectrogram"]],  range_callback,   False],
          ["mel_dct",         "mel & DCT",               '',               '7,7',                  1, ["representation",
                                                                                                       ["mel-cepstrum"]], mel_dct_callback, True],
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
    ndown = int(model_parameters['ndown'])
    nup = int(model_parameters['nup'])
    nconv = int(model_parameters['nconv'])
    nfilters = int(model_parameters['nfilters'])
    kernel_size = int(model_parameters['kernel_size'])
    padding = model_parameters['padding']
    dropout = float(model_parameters['dropout'])/100
    #hyperparameter1 = int(model_parameters["my-simple-textbox"])
    #nonnegative = int(model_parameters["a-bounded-value"])

    # hidden_layers is used to visualize intermediate clusters in the GUI
    hidden_layers = []

    window_tics = stride_tics = 1

    # 'parallelize' specifies the number of output tics to evaluate
    # simultaneously when classifying.  stride (from e.g. spectrograms)
    # and downsampling (from e.g. conv kernel strides) must be taken into
    # account to get the corresponding number of input tics
    context_tics = int(model_settings['audio_tic_rate'] * model_settings['context'] * model_settings['time_scale'])
    ninput_tics = context_tics + (model_settings['parallelize']-1) * stride_tics * 2**(ndown-nup)
    noutput_tics = (context_tics-window_tics) // stride_tics + 1
    input_layer = Input(shape=(ninput_tics, model_settings["audio_nchannels"]))
    hidden_layers.append(input_layer)
    x = input_layer

    # adapted from https://github.com/krentzd/sparse-unet/blob/main/model.py
    # to make it look like https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/
    def down_block_1D(x, nfilters):
        nonlocal noutput_tics
        for _ in range(nconv):
            x = Conv1D(filters=nfilters, kernel_size=kernel_size, padding=padding)(x)
            hidden_layers.append(x)
            x = BatchNormalization()(x)
            x = ReLU()(x)
            if padding == 'valid':
                noutput_tics = noutput_tics - kernel_size + 1

        return x

    def up_block_1D(x, concat_layer, nfilters):
        nonlocal noutput_tics
        x = Conv1DTranspose(nfilters, kernel_size=2, strides=2)(x)
        noutput_tics = math.ceil(noutput_tics * 2 - 1 + 2 - 1)   ### ???

        x_shape = x.shape
        concat_shape = concat_layer.shape
        hoffset = (concat_shape[1] - x_shape[1]) // 2
        #woffset = (concat_shape[2] - x_shape[2]) // 2
        #x = Concatenate()([x, Slice([0,hoffset,woffset,0],
        #                            [-1,x_shape[1],x_shape[2],-1])(concat_layer)])
        x = Concatenate()([x, Slice([0,hoffset,0],
                                    [-1,x_shape[1],-1])(concat_layer)])

        for _ in range(nconv):
            x = Conv1D(filters=nfilters, kernel_size=kernel_size, padding=padding)(x)
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
    x = down_block_1D(x, nfilters=nfilters*(2**ndown))

    if dropout>0:
        x = Dropout(dropout)(x)

    for i,_ in enumerate(range(nup)):
        x = up_block_1D(x, ds.pop(), nfilters=nfilters*(2**(ndown-i-1)))

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
