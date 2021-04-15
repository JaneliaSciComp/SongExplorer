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
import tensorflow as tf

def prepare_model_settings(nlabels, audio_tic_rate, nchannels, 
                           nwindows, batch_size,
                           context_ms, representation,
                           window_ms, stride_ms,
                           dct_ncoefficients, filterbank_nchannels,
                           model_parameters):
  """Calculates common settings needed for all models.

  Args:
    nlabels: How many classes are to be recognized.
    audio_tic_rate: Number of audio tics per second.
    context_ms: Length of each audio clip to be analyzed.
    window_ms: Duration of frequency analysis window.
    stride_ms: How far to move in time between frequency windows.
    dct_ncoefficients: Number of frequency bins to use for analysis.

  Returns:
    Dictionary containing common settings.
  """
  context_tics = int(audio_tic_rate * context_ms / 1000)
  window_tics = int(audio_tic_rate * window_ms / 1000)
  stride_tics = int(audio_tic_rate * stride_ms / 1000)
  length_minus_window = (context_tics - window_tics)
  if length_minus_window < 0:
    spectrogram_length = 0
  else:
    spectrogram_length = 1 + int(length_minus_window / stride_tics)
  if representation=='waveform':
    input_nfreqs = 1
    input_ntimes = context_tics
  elif representation=='spectrogram':
    input_nfreqs = window_tics//2+1
    input_ntimes = spectrogram_length
  elif representation=='mel-cepstrum':
    input_nfreqs = dct_ncoefficients
    input_ntimes = spectrogram_length
  print('context_tics = %d' % (context_tics))
  print('nchannels = %d' % (nchannels))
  print('window_tics = %d' % (window_tics))
  print('stride_tics = %d' % (stride_tics))
  return {**{'context_tics': context_tics,
             'nchannels': nchannels,
             'representation': representation,
             'window_tics': window_tics,
             'stride_tics': stride_tics,
             'input_nfreqs': input_nfreqs,  ###  and fingerprint
             'input_ntimes': input_ntimes,
             'nwindows': nwindows,
             'dct_ncoefficients': dct_ncoefficients,
             'filterbank_nchannels': filterbank_nchannels,
             'nlabels': nlabels,
             'audio_tic_rate': audio_tic_rate,
             'batch_size': batch_size,
            },
          **model_parameters}
