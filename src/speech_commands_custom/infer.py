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
r"""
run just the forward pass of model on the test set
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path
import sys

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import input_data
import models
from tensorflow.python.platform import gfile

from tensorflow.python.platform import tf_logging as logging
logging.get_logger().propagate = False

import datetime as dt

import json

import importlib

FLAGS = None


def main(_):
  sys.path.append(os.path.dirname(FLAGS.model_architecture))
  model = importlib.import_module(os.path.basename(FLAGS.model_architecture))

  # We want to see all the logging messages for this tutorial.
  tf.logging.set_verbosity(tf.logging.INFO)
  np.set_printoptions(threshold=np.inf,linewidth=10000)

  flags = vars(FLAGS)
  for key in sorted(flags.keys()):
    tf.logging.info('%s = %s', key, flags[key])

  # Start a new TensorFlow session.
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  config.allow_soft_placement = True
  #config.log_device_placement = False
  sess = tf.InteractiveSession(config=config)

  label_file = os.path.join(os.path.dirname(FLAGS.start_checkpoint), "labels.txt")
  fid = open(label_file)
  labels = []
  for line in fid:
    labels.append(line.rstrip())
  label_count = len(labels)
  fid.close()

  model_settings = models.prepare_model_settings(
      label_count,
      FLAGS.sample_rate,
      FLAGS.nchannels,
      1,
      FLAGS.batch_size,
      FLAGS.clip_duration_ms,
      FLAGS.representation,
      FLAGS.window_size_ms, FLAGS.window_stride_ms,
      FLAGS.dct_coefficient_count, FLAGS.filterbank_channel_count,
      FLAGS.model_parameters)

  fingerprint_size = model_settings['fingerprint_size']
  time_shift_samples = int((FLAGS.time_shift_ms * FLAGS.sample_rate) / 1000)

  fingerprint_input = tf.placeholder(
      tf.float32, [None, fingerprint_size], name='fingerprint_input')

  hidden, logits = model.create_model(
      fingerprint_input,
      model_settings,
      is_training=False)

  tf.global_variables_initializer().run()

  models.load_variables_from_checkpoint(sess, FLAGS.start_checkpoint)

  total_parameters = 0
  for variable in tf.trainable_variables():
      shape = variable.get_shape()
      variable_parameters = 1
      for dim in shape:
          variable_parameters *= int(dim)
      total_parameters += variable_parameters
  tf.logging.info('number of trainable parameters: %d',total_parameters)

  audio_processor = input_data.AudioProcessor(
      FLAGS.data_url, FLAGS.data_dir,
      FLAGS.time_shift_ms, FLAGS.time_shift_random,
      FLAGS.wanted_words.split(','), FLAGS.labels_touse.split(','),
      FLAGS.validation_percentage, FLAGS.validation_offset_percentage,
      FLAGS.validation_files.split(','),
      FLAGS.testing_percentage, FLAGS.testing_files.split(','), FLAGS.subsample_skip,
      FLAGS.subsample_word,
      FLAGS.partition_word, FLAGS.partition_n, FLAGS.partition_training_files.split(','),
      FLAGS.partition_validation_files.split(','),
      -1,
      FLAGS.testing_equalize_ratio, FLAGS.testing_max_samples,
      model_settings)

  testing_set_size = audio_processor.set_size('testing')

  for isample in xrange(0, testing_set_size, FLAGS.batch_size):
    fingerprints, _, samples = (
        audio_processor.get_data(FLAGS.batch_size, isample, model_settings,
                                 0.0 if FLAGS.time_shift_random else time_shift_samples,
                                 FLAGS.time_shift_random,
                                 'testing', sess))
    needed = FLAGS.batch_size - fingerprints.shape[0]
    if needed>0:
      fingerprints = np.append(fingerprints,
            np.repeat(fingerprints[[0],:],needed,axis=0), axis=0)
      for _ in range(needed):
        samples.append(samples[0])
    logit_vals, hidden_vals = sess.run(
        [logits, hidden],
        feed_dict={
            fingerprint_input: fingerprints,
        })
    batch_size = min(FLAGS.batch_size, testing_set_size - isample)
    obtained = FLAGS.batch_size - needed
    if isample==0:
      samples_data = [None]*testing_set_size
    samples_data[isample:isample+obtained] = samples[:obtained]
    if FLAGS.save_activations:
      if isample==0:
        activations = []
        for ihidden in range(len(hidden_vals)):
          nHWC = np.shape(hidden_vals[ihidden])[1:]
          activations.append(np.empty((testing_set_size, *nHWC)))
        activations.append(np.empty((testing_set_size, np.shape(logit_vals)[1])))
      for ihidden in range(len(hidden_vals)):
        activations[ihidden][isample:isample+obtained,:,:] = \
              hidden_vals[ihidden][:obtained,:,:,:]
      activations[-1][isample:isample+obtained,:] = logit_vals[:obtained,:]
    if FLAGS.save_fingerprints:
      if isample==0:
        nW = round((FLAGS.clip_duration_ms - FLAGS.window_size_ms) / \
                   FLAGS.window_stride_ms + 1)
        nH = round(np.shape(fingerprints)[1]/nW)
        input_layer = np.empty((testing_set_size,nW,nH))
      input_layer[isample:isample+obtained,:,:] = \
            np.reshape(fingerprints[:obtained,:],(obtained,nW,nH))
  if FLAGS.save_activations:
    np.savez(os.path.join(FLAGS.data_dir,'activations.npz'), \
             *activations, samples=samples_data, labels=labels)
  if FLAGS.save_fingerprints:
    np.save(os.path.join(FLAGS.data_dir,'fingerprints.npy'), input_layer)

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--data_url',
      type=str,
      # pylint: disable=line-too-long
      default='http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz',
      # pylint: enable=line-too-long
      help='Location of speech training data archive on the web.')
  parser.add_argument(
      '--data_dir',
      type=str,
      default='/tmp/speech_dataset/',
      help="""\
      Where to download the speech training data to.
      """)
  parser.add_argument(
      '--time_shift_ms',
      type=float,
      default=100.0,
      help="""\
      Range to shift the training audio by in time.
      """)
  parser.add_argument(
      '--time_shift_random',
      type=str2bool,
      default=True,
      help="""\
      True shifts randomly within +/- time_shift_ms; False shifts by exactly time_shift_ms.
      """)
  parser.add_argument(
      '--testing_percentage',
      type=float,
      default=10,
      help='What percentage of wavs to use as a test set.')
  parser.add_argument(
      '--testing_files',
      type=str,
      default='',
      help='Which wav files to use as a test set.')
  parser.add_argument(
      '--subsample_word',
      type=str,
      default='',
      help='Train on only a subset of annotations for this word.')
  parser.add_argument(
      '--subsample_skip',
      type=str,
      default='',
      help='Take only every Nth annotation for the specified word.')
  parser.add_argument(
      '--partition_word',
      type=str,
      default='',
      help='Train on only a fixed number of annotations for this word.')
  parser.add_argument(
      '--partition_n',
      type=int,
      default=0,
      help='Train on only this number of annotations from each file for the specified word.')
  parser.add_argument(
      '--partition_training_files',
      type=str,
      default='',
      help='Train on only these files for the specified word.')
  parser.add_argument(
      '--partition_validation_files',
      type=str,
      default='',
      help='Validate on only these files for the specified word.')
  parser.add_argument(
      '--validation_files',
      type=str,
      default='',
      help='Which wav files to use as a validation set.')
  parser.add_argument(
      '--validation_percentage',
      type=float,
      default=10,
      help='What percentage of wavs to use as a validation set.')
  parser.add_argument(
      '--validation_offset_percentage',
      type=float,
      default=0,
      help='Which wavs to use as a cross-validation set.')
  parser.add_argument(
      '--sample_rate',
      type=int,
      default=16000,
      help='Expected sample rate of the wavs',)
  parser.add_argument(
      '--nchannels',
      type=int,
      default=1,
      help='Expected number of channels in the wavs',)
  parser.add_argument(
      '--clip_duration_ms',
      type=float,
      default=1000,
      help='Expected duration in milliseconds of the wavs',)
  parser.add_argument(
      '--window_size_ms',
      type=float,
      default=30.0,
      help='How long each spectrogram timeslice is.',)
  parser.add_argument(
      '--window_stride_ms',
      type=float,
      default=10.0,
      help='How far to move in time between spectogram timeslices.',)
  parser.add_argument(
      '--filterbank_channel_count',
      type=int,
      default=40,
      help='How many internal bins to use for the MFCC fingerprint',)
  parser.add_argument(
      '--dct_coefficient_count',
      type=int,
      default=40,
      help='How many output bins to use for the MFCC fingerprint',)
  parser.add_argument(
      '--batch_size',
      type=int,
      default=100,
      help='How many items to train with at once',)
  parser.add_argument(
      '--wanted_words',
      type=str,
      default='yes,no,up,down,left,right,on,off,stop,go',
      help='Words to use (others will be added to an unknown label)',)
  parser.add_argument(
      '--labels_touse',
      type=str,
      default='annotated,classified',
      help='A comma-separted list of "annotated", "detected" , or "classified"',)
  parser.add_argument(
      '--start_checkpoint',
      type=str,
      default='',
      help='If specified, restore this pretrained model before any training.')
  parser.add_argument(
      '--testing_equalize_ratio',
      type=int,
      default=0,
      help='Limit most common word to be no more than this times more than the least common word for testing.')
  parser.add_argument(
      '--testing_max_samples',
      type=int,
      default=0,
      help='Limit number of test samples to this number.')
  parser.add_argument(
      '--representation',
      type=str,
      default='waveform',
      help='What input representation to use.  One of waveform, spectrogram, or mel-cepstrum.')
  parser.add_argument(
      '--optimizer',
      type=str,
      default='sgd',
      help='What optimizer to use.  One of sgd, adam, adagrad, or rmsprop.')
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
      '--save_activations',
      type=str2bool,
      default=False,
      help='Whether to save hidden layer activations during processing')
  parser.add_argument(
      '--save_fingerprints',
      type=str2bool,
      default=False,
      help='Whether to save fingerprint input layer during processing')

  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
