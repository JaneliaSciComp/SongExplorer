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
r"""Converts a trained checkpoint into a frozen model for mobile inference.

Once you've trained a model using the `train.py` script, you can use this tool
to convert it into a binary GraphDef file that can be loaded into the Android,
iOS, or Raspberry Pi example code. Here's an example of how to run it:

bazel run tensorflow/examples/speech_commands/freeze -- \
--audio_tic_rate=16000 \
--context_ms=1000 \
--model_architecture=conv \
--start_checkpoint=/tmp/speech_commands_train/conv.ckpt-1300 \
--output_file=/tmp/my_frozen_graph.pb

One thing to watch out for is that you need to pass in the same arguments for
`audio_tic_rate` and other command line variables here as you did for the training
script.

The resulting graph has an input for WAV-encoded data named 'wav_data', one for
raw PCM data (as floats in the range -1.0 to 1.0) called 'decoded_sound_data',
and the output is called 'labels_softmax'.

"""
import argparse
import os.path
import sys

import tensorflow as tf

import models

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

  labels_list = FLAGS.labels_touse.split(',')
  model_settings = models.prepare_model_settings(
      len(labels_list), FLAGS.audio_tic_rate, FLAGS.nchannels,
      FLAGS.parallelize, 1, FLAGS.context_ms,
      FLAGS.model_parameters)

  thismodel = model.create_model(model_settings)
  thismodel.summary()

  checkpoint = tf.train.Checkpoint(thismodel=thismodel)
  checkpoint.read(FLAGS.start_checkpoint).expect_partial()

  class InferenceStep(tf.Module):
      def __init__(self, thismodel):
          self.thismodel = thismodel
          self.input_shape = thismodel.input_shape

      @tf.function(input_signature=[])
      def get_input_shape(self):
          return self.input_shape

      @tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32)])
      def inference_step(self, waveform):
          hidden, output = self.thismodel(waveform, training=False)
          return hidden, tf.nn.softmax(output)

  return InferenceStep(thismodel)


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
      '--nchannels',
      type=int,
      default=1,
      help='Expected number of channels in the wavs',)
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
  main()
