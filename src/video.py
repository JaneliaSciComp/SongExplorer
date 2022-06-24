# modified from https://github.com/salinasJJ/BBaction
#
# MIT License
# 
# Copyright (c) 2021 BB-Repos
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import logging 
bokehlog = logging.getLogger("songexplorer") 

import os
import sys
import importlib

use_audio=0
use_video=1

model_parameters = [
  # key in `model_settings`, title in GUI, '' for textbox or [] for pull-down, default value, enable logic, callback, required
  ["use_bias",       "use bias",       ["yes", "no"],              "yes",         [], None, True],
  ["augmentation",   "augmentation",   ['none',
                                        'flip',
                                        'rotate',
                                        'both'],                   'both',        [], None, True],
  ["initializer",    "initializer",    ['he_normal',
                                        'glorot_normal',
                                        'lecun_normal',
                                        'random_normal',
                                        'truncated_normal'],       'he_normal',   [], None, True],
  ["dropout_rate",   "dropout %",      '',                         '50',          [], None, True],
  ["depth",          "# layers",       ['13', '26', '50', '101', '152'], '26',          [], None, True],
  ["bn_momentum",    "BN momentum",    '',                         '0.9',         [], None, True],
  ["epsilon",        "BN epsilon",     '',                         '0.0001',      [], None, True],
  ["arch",           "architecture",   ['ip-csn',
                                       'ir-csn',
                                       'ip'],                      'ip-csn',       [], None, True],
  ["regularization", "regularization", ['weight_decay',
                                        'l2'],                     'weight_decay', [], None, True],
  ]

import math

import tensorflow as tf
from tensorflow.keras import layers, regularizers

import numpy as np

NUM_BLOCKS = {
    13: (1, 1, 1, 1),
    26: (2, 2, 2, 2),
    50: (3, 4, 6, 3),
    101: (3, 4, 23, 3),
    152: (3, 8, 36, 3),
}
FILTERS = [256, 512, 1024, 2048]
STRIDES = [
    (1, 1, 1),
    (1, 2, 2),
    (1, 2, 2),
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

def regularizer(PARAMS):
    if PARAMS['regularization'] == 'weight_decay':
        return None
    elif PARAMS['regularization'] == 'l2':
        return regularizers.l2(PARAMS['weight_decay'])

def update_noutput_tics(PARAMS, kernel_size, stride):
    PARAMS["noutput_tics"] = math.ceil((PARAMS["noutput_tics"]-kernel_size+1) / stride)

def block(PARAMS, inputs, stride_idx):
    FILTER_DIM = 1 if PARAMS['data_format'] == 'channels_first' else -1
    kernel_size_touse=[3,3,3] if PARAMS["noutput_tics"]>=3 else [1,3,3]
    if PARAMS['arch'] in ['ir-csn', 'ir']:
        x = inputs
        x = layers.Conv3D(
            filters=x.shape[FILTER_DIM],
            kernel_size=kernel_size_touse,
            strides=STRIDES[stride_idx],
            use_bias=PARAMS['use_bias'],
            kernel_initializer=PARAMS['initializer'],
            kernel_regularizer=regularizer(PARAMS),
            data_format=PARAMS['data_format'],
            groups=x.shape[FILTER_DIM],
            )(x)
        kernels.append(3)
        strides.append(STRIDES[stride_idx][1])
        update_noutput_tics(PARAMS, kernel_size_touse[0], STRIDES[stride_idx][0])
    elif PARAMS['arch'] in ['ip-csn', 'ip']:
        x = layers.Conv3D(
            filters=inputs.shape[FILTER_DIM],
            kernel_size=[1,1,1],
            strides=[1,1,1],
            use_bias=PARAMS['use_bias'],
            kernel_initializer=PARAMS['initializer'],
            kernel_regularizer=regularizer(PARAMS),
            data_format=PARAMS['data_format'],
            )(inputs)
        kernels.append(1)
        strides.append(1)
        update_noutput_tics(PARAMS, 1, 1)
        x = layers.BatchNormalization(
            momentum=PARAMS['bn_momentum'],
            epsilon=PARAMS['epsilon'],
            scale=False,
            gamma_regularizer=regularizer(PARAMS),
            beta_regularizer=regularizer(PARAMS),
            )(x)
        x = layers.ReLU()(x)

        x = layers.Conv3D(
            filters=x.shape[FILTER_DIM],
            kernel_size=kernel_size_touse,
            strides=STRIDES[stride_idx],
            use_bias=PARAMS['use_bias'],
            kernel_initializer=PARAMS['initializer'],
            kernel_regularizer=regularizer(PARAMS),
            data_format=PARAMS['data_format'],
            groups=x.shape[FILTER_DIM],
            )(x)
        kernels.append(3)
        strides.append(STRIDES[stride_idx][1])
        update_noutput_tics(PARAMS, kernel_size_touse[0], STRIDES[stride_idx][0])
    return x
        
def bottleneck(PARAMS,
        inputs, 
        filter_idx,
        stride_idx
    ):
    FILTER_DIM = 1 if PARAMS['data_format'] == 'channels_first' else -1
    num_filters = FILTERS[filter_idx] \
                        if PARAMS['depth'] >= 50 else FILTERS[filter_idx] / 4
    skip = inputs

    x = layers.Conv3D(
        filters=int(num_filters / 4) if PARAMS['depth'] >= 50 else num_filters,
        kernel_size=[1,1,1],
        strides=[1,1,1],
        use_bias=PARAMS['use_bias'],
        kernel_initializer=PARAMS['initializer'],
        kernel_regularizer=regularizer(PARAMS),
        data_format=PARAMS['data_format'],
        )(inputs)
    kernels.append(1)
    strides.append(1)
    update_noutput_tics(PARAMS, 1, 1)
    x = layers.BatchNormalization(
        momentum=PARAMS['bn_momentum'],
        epsilon=PARAMS['epsilon'],
        scale=False,
        gamma_regularizer=regularizer(PARAMS),
        beta_regularizer=regularizer(PARAMS),
        )(x)
    x = layers.ReLU()(x)

    x = block(PARAMS,
        x, 
        stride_idx=stride_idx,
    )
    x = layers.BatchNormalization(
        momentum=PARAMS['bn_momentum'],
        epsilon=PARAMS['epsilon'],
        scale=False,
        gamma_regularizer=regularizer(PARAMS),
        beta_regularizer=regularizer(PARAMS),
        )(x)
    x = layers.ReLU()(x)
    hidden_layers.append(x)
    
    x = layers.Conv3D(
        filters=num_filters,
        kernel_size=[1,1,1],
        strides=[1,1,1],
        use_bias=PARAMS['use_bias'],
        kernel_initializer=PARAMS['initializer'],
        kernel_regularizer=regularizer(PARAMS),
        data_format=PARAMS['data_format'],
        )(x)
    kernels.append(1)
    strides.append(1)
    update_noutput_tics(PARAMS, 1, 1)
    x = layers.BatchNormalization(
        momentum=PARAMS['bn_momentum'],
        epsilon=PARAMS['epsilon'],
        scale=False,
        gamma_regularizer=regularizer(PARAMS),
        beta_regularizer=regularizer(PARAMS),
        )(x)

    if skip.shape[FILTER_DIM] != num_filters:
        skip = layers.Conv3D(
            filters=num_filters,
            kernel_size=[1,1,1],
            strides=STRIDES[stride_idx],
            use_bias=PARAMS['use_bias'],
            kernel_initializer=PARAMS['initializer'],
            kernel_regularizer=regularizer(PARAMS),
            data_format=PARAMS['data_format'],
            )(skip)
        kernels.append(1)
        strides.append(STRIDES[stride_idx][1])
        update_noutput_tics(PARAMS, 1, STRIDES[stride_idx][0])
        skip = layers.BatchNormalization(
            momentum=PARAMS['bn_momentum'],
            epsilon=PARAMS['epsilon'],
            scale=False,
            gamma_regularizer=regularizer(PARAMS),
            beta_regularizer=regularizer(PARAMS),
            )(skip)
    if skip.shape != x.shape:
        skip = Slice([0,skip.shape[1]//4,1,1,0], [-1,x.shape[1],x.shape[2],x.shape[3],-1])(skip)
    x = layers.Add()([
        skip, 
        x
    ])
    x = layers.ReLU()(x)
    return x
    
def network(PARAMS, inputs):
    x = inputs
    x = layers.Conv3D(
        filters=64, 
        kernel_size=(1,7,7), 
        strides=(1,2,2), 
        use_bias=PARAMS['use_bias'],
        kernel_initializer=PARAMS['initializer'],
        kernel_regularizer=regularizer(PARAMS),
        data_format=PARAMS['data_format'],
        )(x)
    kernels.append(7)
    strides.append(2)
    update_noutput_tics(PARAMS, 1, 1)
    x = layers.BatchNormalization(
        momentum=PARAMS['bn_momentum'],
        epsilon=PARAMS['epsilon'],
        scale=False,
        gamma_regularizer=regularizer(PARAMS),
        beta_regularizer=regularizer(PARAMS),
        )(x)
    x = layers.ReLU()(x)
    
    for _ in tf.range(NUM_BLOCKS[PARAMS['depth']][0]):
        x = bottleneck(PARAMS,
            x, 
            filter_idx=tf.constant(0),
            stride_idx=tf.constant(0),
        )
    
    x = bottleneck(PARAMS,
        x, 
        filter_idx=tf.constant(1), 
        stride_idx=tf.constant(1),
    )
    for _ in tf.range(NUM_BLOCKS[PARAMS['depth']][1] - 1):
        x = bottleneck(PARAMS,
            x, 
            filter_idx=tf.constant(1),
            stride_idx=tf.constant(0),
        )
    
    if PARAMS['context_frames'] < 4:
        x = bottleneck(PARAMS,
            x, 
            filter_idx=tf.constant(2), 
            stride_idx=tf.constant(2),
        )
    else:
        x = bottleneck(PARAMS,
            x, 
            filter_idx=tf.constant(2),
            stride_idx=tf.constant(1),
        )
    for _ in tf.range(NUM_BLOCKS[PARAMS['depth']][2] - 1):
        x = bottleneck(PARAMS,
            x, 
            filter_idx=tf.constant(2),
            stride_idx=tf.constant(0),
        )
   
    if PARAMS['context_frames'] < 8:
        x = bottleneck(PARAMS,
            x, 
            filter_idx=tf.constant(3),
            stride_idx=tf.constant(2),
        )
    else:
        x = bottleneck(PARAMS,
            x, 
            filter_idx=tf.constant(3), 
            stride_idx=tf.constant(1),
        )
    for _ in tf.range(NUM_BLOCKS[PARAMS['depth']][3] - 1):
        x = bottleneck(PARAMS,
            x, 
            filter_idx=tf.constant(3),
            stride_idx=tf.constant(0),
        )

    # https://distill.pub/2019/computing-receptive-fields/
    print("receptive_field = %d" % int(1+np.sum((np.array(kernels)-1)*np.cumprod(strides[:-1]))))

    x = layers.AveragePooling3D(
        pool_size=[
            #PARAMS['context_frames'] // 8 \
            #    if PARAMS['context_frames'] >= 8 else 1, 
            1, 
            11, 
            11,
        ], 
        strides=[1, 1, 1],
        data_format=PARAMS['data_format'],
        )(x)

    x = layers.Dropout(PARAMS['dropout_rate'])(x)
    i = PARAMS['data_format'] == 'channels_first'
    x = layers.Conv3D(
        filters=PARAMS['num_labels'],
        kernel_size=[PARAMS["noutput_tics"], *x.get_shape().as_list()[(i+2):(i+4)]],
        use_bias=PARAMS['use_bias'],
        kernel_initializer=PARAMS['initializer'],
        kernel_regularizer=regularizer(PARAMS),
        data_format=PARAMS['data_format'],
        )(x)
    kernels.append(x.get_shape().as_list()[i+3])
    strides.append(1)
    x = layers.Activation(
        'linear', 
        dtype='float32',
        )(x)
    x = layers.Reshape((-1,PARAMS['num_labels']))(x)
    return x

hidden_layers = []
kernels = []
strides = [1]

def get_model(PARAMS):
    PARAMS["noutput_tics"] = PARAMS['context_frames']
    if PARAMS['data_format'] == 'channels_first':
        inputs = tf.keras.Input([
            PARAMS['nchannels'],
            PARAMS['clip_frames'],
            PARAMS['spatial_height'],
            PARAMS['spatial_width'],
        ])
    elif PARAMS['data_format'] == 'channels_last':
        inputs = tf.keras.Input([
            PARAMS['clip_frames'],
            PARAMS['spatial_height'],
            PARAMS['spatial_width'],
            PARAMS['nchannels'],
        ])

    x = inputs
    x = tf.squeeze(inputs, axis=4)  # assumes grayscale
    x = tf.transpose(x, perm=[0,2,3,1])
   
    if PARAMS['augmentation']=='flip' or PARAMS['augmentation']=='both':
        x = layers.RandomFlip()(x)
    if PARAMS['augmentation']=='rotate' or PARAMS['augmentation']=='both':
        x = layers.RandomRotation(0.5, fill_mode='constant', fill_value=0)(x)

    x = tf.transpose(x, perm=[0,3,1,2])
    x = tf.expand_dims(x, axis=4)

    outputs = network(PARAMS, x)

    return tf.keras.Model(inputs, outputs = [hidden_layers, outputs])

def create_model(model_settings):
  downsample_by = 1
  params = model_settings.copy()
  params['context_frames'] = round(params['context_ms'] / 1000 * params['video_frame_rate'])
  params['clip_frames'] = params['context_frames']+(params["parallelize"]-1)*downsample_by
  params['spatial_height'] = int(params['video_frame_height'])
  params['spatial_width'] = int(params['video_frame_width'])
  params['num_labels'] = int(params['nlabels'])
  params['nchannels'] = len(params['video_channels'])
  params['use_bias'] = params['use_bias']=="yes"
  params['data_format'] = 'channels_last'
  params['dropout_rate'] = float(params['dropout_rate'])/100
  params['depth'] = int(params['depth'])
  params['bn_momentum'] = float(params['bn_momentum'])
  params['epsilon'] = float(params['epsilon'])

  return get_model(params)
