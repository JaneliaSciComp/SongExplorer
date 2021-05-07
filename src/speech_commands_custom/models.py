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
def prepare_model_settings(nlabels, audio_tic_rate, nchannels, 
                           parallelize, batch_size,
                           context_ms,
                           model_parameters):
  """Calculates common settings needed for all models.

  Args:
    nlabels: How many classes are to be recognized.
    audio_tic_rate: Number of audio tics per second.
    context_ms: Length of each audio clip to be analyzed.

  Returns:
    Dictionary containing common settings.
  """
  context_tics = int(audio_tic_rate * context_ms / 1000)
  print('context_tics = %d' % (context_tics))
  print('nchannels = %d' % (nchannels))
  return {**{'context_tics': context_tics,
             'nchannels': nchannels,
             'parallelize': parallelize,
             'nlabels': nlabels,
             'audio_tic_rate': audio_tic_rate,
             'batch_size': batch_size,
            },
          **model_parameters}
