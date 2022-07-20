#!/usr/bin/env python3

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


# save a trained model into a binary file that can be used as a classifier

# e.g.
# $SONGEXPLORER_BIN freeze.py \
#      --context_ms=204.8 \
#      --model_architecture=convolutional \
#      --model_parameters='{"representation":"waveform", "window_ms":6.4, "stride_ms":1.6, "mel_dct":"7,7", "dropout":0.5, "kernel_sizes":5,3,3", last_conv_width":130, "nfeatures":"256,256,256", "dilate_after_layer":65535, "stride_after_layer":65535, "connection_type":"plain"}' \
#      --start_checkpoint=`pwd`/trained-classifier/train_1k/ckpt-50 \
#      --output_file=`pwd`/trained-classifier/train_1k/frozen-graph.ckpt-50.pb \
#      --labels_touse=pulse,sine,ambient \
#      --parallelize=16384 \
#      --audio_tic_rate=5000 \
#      --audio_nchannels=1


import argparse
import os.path
import sys

from datetime import datetime
import socket

import tensorflow as tf
import json
import importlib

FLAGS = None

def create_inference_graph():
  """Creates an audio model with the nodes needed for inference.

  Uses the supplied arguments to create a model, and inserts the input and
  output nodes that are needed to use the graph for inference.

  Args:
    labels_touse: Comma-separated list of the labels we're trying to recognize.
    audio_tic_rate: How many tics per second are in the input audio files.
    context_ms: How many tics to analyze for the audio pattern.
    clip_stride_ms: How often to run recognition. Useful for models with cache.
    model_architecture: Name of the kind of model to generate.
  """
  sys.path.append(os.path.dirname(FLAGS.model_architecture))
  model = importlib.import_module(os.path.basename(FLAGS.model_architecture))

  model_settings = {'nlabels': len(FLAGS.labels_touse.split(',')),
                    'audio_tic_rate': FLAGS.audio_tic_rate,
                    'audio_nchannels': FLAGS.audio_nchannels,
                    'video_frame_rate': FLAGS.video_frame_rate,
                    'video_frame_width': FLAGS.video_frame_width,
                    'video_frame_height': FLAGS.video_frame_height,
                    'video_channels': [int(x)-1 for x in FLAGS.video_channels.split(',')],
                    'parallelize': FLAGS.parallelize,
                    'batch_size': 1,
                    'context_ms': FLAGS.context_ms }
  thismodel = model.create_model(model_settings, FLAGS.model_parameters)
  thismodel.summary()

  checkpoint = tf.train.Checkpoint(thismodel=thismodel)
  checkpoint.read(FLAGS.start_checkpoint).expect_partial()

  class InferenceStep(tf.Module):
      def __init__(self, model, thismodel):
          self.thismodel = thismodel
          self.input_shape = thismodel.input_shape
          self.use_audio = model.use_audio
          self.use_video = model.use_video

      @tf.function(input_signature=[])
      def get_input_shape(self):
          return self.input_shape

      @tf.function(input_signature=[])
      def get_use_audio(self):
          return self.use_audio

      @tf.function(input_signature=[])
      def get_use_video(self):
          return self.use_video

      if model.use_audio and model.use_video:
          signature = [[tf.TensorSpec(shape=None, dtype=tf.float32),
                        tf.TensorSpec(shape=None, dtype=tf.float32)]]
      else:
          signature = [tf.TensorSpec(shape=None, dtype=tf.float32)]

      @tf.function(input_signature=signature)
      def inference_step(self, waveform):
          hidden, output = self.thismodel(waveform, training=False)
          return hidden, tf.nn.softmax(output)

  return InferenceStep(model, thismodel)


def main():
  flags = vars(FLAGS)
  for key in sorted(flags.keys()):
    print('%s = %s' % (key, flags[key]))

  # Create the model and load its weights.
  thismodel = create_inference_graph()

  tf.saved_model.save(thismodel, FLAGS.output_file+'/')
  print('Saved frozen graph to %s' % FLAGS.output_file)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--audio_tic_rate',
      type=int,
      default=16000,
      help='Expected tic rate of the wavs',)
  parser.add_argument(
      '--audio_nchannels',
      type=int,
      default=1,
      help='Expected number of channels in the wavs',)
  parser.add_argument(
      '--video_frame_rate',
      type=int,
      default=0,
      help='Expected frame rate in Hz of the video',)
  parser.add_argument(
      '--video_frame_width',
      type=int,
      default=0,
      help='Expected frame width in pixels of the video',)
  parser.add_argument(
      '--video_frame_height',
      type=int,
      default=0,
      help='Expected frame height in pixels of the video',)
  parser.add_argument(
      '--video_channels',
      type=str,
      default='1',
      help='Comma-separated list of which channels in the video to use',)
  parser.add_argument(
      '--context_ms',
      type=float,
      default=1000,
      help='Expected duration in milliseconds of the wavs',)
  parser.add_argument(
      '--parallelize',
      type=int,
      default=1,
      help='How many context windows to process simultaneously',)
  parser.add_argument(
      '--start_checkpoint',
      type=str,
      default='',
      help='If specified, restore this pretrained model before any training.')
  parser.add_argument(
      '--model_architecture',
      type=str,
      default='conv',
      help='What model architecture to use')
  parser.add_argument(
      '--model_parameters',
      type=json.loads,
      default='{}',
      help='What model parameters to use')
  parser.add_argument(
      '--labels_touse',
      type=str,
      default='yes,no,up,down,left,right,on,off,stop,go',
      help='Words to use (others will be added to an unknown label)',)
  parser.add_argument(
      '--output_file', type=str, help='Where to save the frozen graph.')
  FLAGS, unparsed = parser.parse_known_args()

  print(str(datetime.now())+": start time")
  repodir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
  with open(os.path.join(repodir, "VERSION.txt"), 'r') as fid:
    print('SongExplorer version = '+fid.read().strip().replace('\n',', '))
  print("hostname = "+socket.gethostname())

  try:
    main()

  except Exception as e:
    print(e)

  finally:
    os.sync()
    print(str(datetime.now())+": finish time")
