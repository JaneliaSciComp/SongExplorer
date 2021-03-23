#This file, originally from the TensorFlow speech recognition tutorial,
#has been heavily modified for use by SongExplorer.


# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Model definitions for simple speech recognition.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

def prepare_model_settings(label_count, sample_rate, nchannels, 
                           nwindows, batch_size,
                           clip_duration_ms, representation,
                           window_size_ms, window_stride_ms,
                           dct_coefficient_count, filterbank_channel_count,
                           model_parameters):
  """Calculates common settings needed for all models.

  Args:
    label_count: How many classes are to be recognized.
    sample_rate: Number of audio samples per second.
    clip_duration_ms: Length of each audio clip to be analyzed.
    window_size_ms: Duration of frequency analysis window.
    window_stride_ms: How far to move in time between frequency windows.
    dct_coefficient_count: Number of frequency bins to use for analysis.

  Returns:
    Dictionary containing common settings.
  """
  desired_samples = int(sample_rate * clip_duration_ms / 1000)
  window_size_samples = int(sample_rate * window_size_ms / 1000)
  window_stride_samples = int(sample_rate * window_stride_ms / 1000)
  length_minus_window = (desired_samples - window_size_samples)
  if length_minus_window < 0:
    spectrogram_length = 0
  else:
    spectrogram_length = 1 + int(length_minus_window / window_stride_samples)
  if representation=='waveform':
    fingerprint_size = desired_samples * nchannels
    input_frequency_size = 1
    input_time_size = desired_samples
  elif representation=='spectrogram':
    fingerprint_size = (window_size_samples//2 + 1) * spectrogram_length * nchannels
    input_frequency_size = window_size_samples//2+1
    input_time_size = spectrogram_length
  elif representation=='mel-cepstrum':
    fingerprint_size = dct_coefficient_count * spectrogram_length * nchannels
    input_frequency_size = dct_coefficient_count
    input_time_size = spectrogram_length
  tf.logging.info('desired_samples = %d' % (desired_samples))
  tf.logging.info('nchannels = %d' % (nchannels))
  tf.logging.info('window_size_samples = %d' % (window_size_samples))
  tf.logging.info('window_stride_samples = %d' % (window_stride_samples))
  return {**{'desired_samples': desired_samples,
             'channel_count': nchannels,
             'representation': representation,
             'window_size_samples': window_size_samples,
             'window_stride_samples': window_stride_samples,
             'input_frequency_size': input_frequency_size,
             'input_time_size': input_time_size,
             'nwindows': nwindows,
             'spectrogram_length': spectrogram_length,
             'dct_coefficient_count': dct_coefficient_count,
             'filterbank_channel_count': filterbank_channel_count,
             'fingerprint_size': fingerprint_size,
             'label_count': label_count,
             'sample_rate': sample_rate,
             'batch_size': batch_size,
            },
          **model_parameters}

def load_variables_from_checkpoint(sess, start_checkpoint):
  """Utility function to centralize checkpoint restoration.

  Args:
    sess: TensorFlow session.
    start_checkpoint: Path to saved checkpoint on disk.
  """
  saver = tf.train.Saver(tf.global_variables())
  saver.restore(sess, start_checkpoint)
