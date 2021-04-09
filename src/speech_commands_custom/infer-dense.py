#This file, originally from the TensorFlow speech recognition tutorial,
#has been heavily modified for use by SongExplorer.


# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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

import argparse
import sys

import numpy as np
import tensorflow as tf

FLAGS = None


def main():
  #Load a wav file and return sample_rate and numpy data of float64 type.
  data, sample_rate = tf.audio.decode_wav(tf.io.read_file(FLAGS.wav))
  sample_rate = sample_rate.numpy()
  print('sample_rate = '+str(sample_rate))

  # Load model and create a tf session to process audio pieces
  thismodel = tf.saved_model.load(FLAGS.model)
  recognize_graph = thismodel.inference_step

  clip_duration_ms = FLAGS.context_ms + FLAGS.stride_ms * (FLAGS.nwindows - 1)
  clip_duration_samples = np.round(clip_duration_ms * sample_rate / 1000).astype(np.int)

  clip_stride_ms = FLAGS.stride_ms * (FLAGS.nwindows - 1)

  data_slice = tf.transpose(data[:clip_duration_samples,:])
  _,outputs = recognize_graph(tf.expand_dims(data_slice, 0))
  downsample_by = int((FLAGS.nwindows-1) / (outputs.shape[1]-1))
  print('downsample_by = '+str(downsample_by))

  window_stride_ms_adjusted = FLAGS.stride_ms * downsample_by
  clip_stride_ms_adjusted = clip_stride_ms + window_stride_ms_adjusted 
  window_stride_samples = np.round(window_stride_ms_adjusted * sample_rate / 1000).astype(np.int)
  clip_stride_samples = np.round(clip_stride_ms_adjusted * sample_rate / 1000).astype(np.int)

  # Inference along audio stream.
  for audio_data_offset in range(0, 1+data.shape[0], clip_stride_samples):
    input_start = audio_data_offset
    input_end = audio_data_offset + clip_duration_samples
    pad_len = input_end - data.shape[0]
    
    data_slice = tf.transpose(data[input_start:input_end,:] if pad_len<=0 else \
                              np.pad(data[input_start:input_end,:],
                                     ((0,pad_len),(0,0)), mode='median'))
    _,outputs = recognize_graph(tf.expand_dims(data_slice, 0))
    current_time_ms = np.round(audio_data_offset * 1000 / sample_rate).astype(np.int)
    if pad_len>0:
      discard_len = np.ceil(pad_len/window_stride_samples).astype(np.int)
      print(str(current_time_ms)+'ms '+
            np.array2string(outputs.numpy()[0,:-discard_len,:],
                            separator=',',
                            threshold=np.iinfo(np.int).max).replace('\n',''))
      break
    else:
      print(str(current_time_ms)+'ms '+
            np.array2string(outputs.numpy()[0,:,:],
                            separator=',',
                            threshold=np.iinfo(np.int).max).replace('\n',''))


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='test_streaming_accuracy')
  parser.add_argument(
      '--wav', type=str, default='', help='The wave file path to evaluate.')
  parser.add_argument(
      '--labels',
      type=str,
      default='',
      help='The label file path containing all possible classes.')
  parser.add_argument(
      '--model', type=str, default='', help='The model used for inference')
  parser.add_argument(
      '--context_ms',
      type=float,
      default=1000,
      help='Length of each audio clip fed into model.')
  parser.add_argument(
      '--stride_ms',
      type=float,
      default=30,
      help='Length of audio clip stride over main trap.')
  parser.add_argument(
      '--nwindows',
      type=int,
      default=1,
      help='')
  parser.add_argument(
      '--verbose',
      action='store_true',
      default=False,
      help='Whether to print streaming accuracy on stdout.')

  FLAGS, unparsed = parser.parse_known_args()
  main()
