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
r"""Simple speech recognition to spot a limited number of keywords.

This is a self-contained example script that will train a very basic audio
recognition model in TensorFlow. It downloads the necessary training data and
runs with reasonable defaults to train within a few hours even only using a CPU.
For more information, please see
https://www.tensorflow.org/tutorials/audio_recognition.

It is intended as an introduction to using neural networks for audio
recognition, and is not a full speech recognition system. For more advanced
speech systems, I recommend looking into Kaldi. This network uses a keyword
detection style to spot discrete words from a small vocabulary, consisting of
"yes", "no", "up", "down", "left", "right", "on", "off", "stop", and "go".

To run the training process, use:

bazel run tensorflow/examples/speech_commands:train

This will write out checkpoints to /tmp/speech_commands_train/, and will
download over 1GB of open source training data, so you'll need enough free space
and a good internet connection. The default data is a collection of thousands of
one-second .wav files, each containing one spoken word. This data set is
collected from https://aiyprojects.withgoogle.com/open_speech_recording, please
consider contributing to help improve this and other models!

As training progresses, it will print out its accuracy metrics, which should
rise above 90% by the end. Once it's complete, you can run the freeze script to
get a binary GraphDef that you can easily deploy on mobile applications.

If you want to train on your own data, you'll need to create .wavs with your
recordings, all at a consistent length, and then arrange them into subfolders
organized by label. For example, here's a possible file structure:

my_wavs >
  up >
    audio_0.wav
    audio_1.wav
  down >
    audio_2.wav
    audio_3.wav
  other>
    audio_4.wav
    audio_5.wav

You'll also need to tell the script what labels to look for, using the
`--wanted_words` argument. In this case, 'up,down' might be what you want, and
the audio in the 'other' folder would be used to train an 'unknown' category.

To pull this all together, you'd run:

bazel run tensorflow/examples/speech_commands:train -- \
--data_dir=my_wavs --wanted_words=up,down

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path
import sys

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import input_data
import models
from tensorflow.python.platform import gfile

import datetime as dt

import json

FLAGS = None


def main(_):
  # We want to see all the logging messages for this tutorial.
  tf.logging.set_verbosity(tf.logging.INFO)
  np.set_printoptions(threshold=np.inf,linewidth=10000)

  flags = vars(FLAGS)
  for key in sorted(flags.keys()):
    tf.logging.info('%s = %s', key, flags[key])

  if FLAGS.random_seed_weights!=-1:
      tf.random.set_random_seed(FLAGS.random_seed_weights)

  # Start a new TensorFlow session.
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  config.allow_soft_placement = True
  #config.log_device_placement = False
  sess = tf.InteractiveSession(config=config)

  # Begin by making sure we have the training data we need. If you already have
  # training data of your own, use `--data_url= ` on the command line to avoid
  # downloading.

  if FLAGS.start_checkpoint!='':
      label_file = os.path.join(os.path.dirname(FLAGS.start_checkpoint), "vgg_labels.txt")
      fid = open(label_file)
      label_count = sum(1 for line in fid)
      fid.close()
  else:
      label_count = len(input_data.prepare_words_list(FLAGS.wanted_words.split(','),
                                                      FLAGS.silence_percentage,
                                                      FLAGS.unknown_percentage))

  model_settings = models.prepare_model_settings(
      label_count,
      FLAGS.sample_rate, FLAGS.clip_duration_ms,
      FLAGS.representation,
      FLAGS.window_size_ms, FLAGS.window_stride_ms, 1,
      FLAGS.dct_coefficient_count, FLAGS.filterbank_channel_count,
      [int(x) for x in FLAGS.filter_counts.split(',')],
      [int(x) for x in FLAGS.filter_sizes.split(',')],
      FLAGS.final_filter_len,
      FLAGS.dropout_prob, FLAGS.batch_size,
      FLAGS.dilate_after_layer, FLAGS.stride_after_layer,
      FLAGS.connection_type)

  audio_processor = input_data.AudioProcessor(
      FLAGS.data_url, FLAGS.data_dir,
      FLAGS.silence_percentage, FLAGS.unknown_percentage,
      FLAGS.wanted_words.split(','), FLAGS.labels_touse.split(','),
      FLAGS.validation_percentage, FLAGS.validation_offset_percentage,
      FLAGS.validation_files.split(','),
      FLAGS.testing_percentage, FLAGS.testing_files.split(','), FLAGS.subsample_skip,
      FLAGS.subsample_word,
      FLAGS.partition_word, FLAGS.partition_n, FLAGS.partition_training_files.split(','),
      FLAGS.partition_validation_files.split(','),
      FLAGS.random_seed_batch,
      FLAGS.testing_equalize_ratio, FLAGS.testing_max_samples,
      model_settings)

  fingerprint_size = model_settings['fingerprint_size']
  label_count = model_settings['label_count']
  time_shift_samples = int((FLAGS.time_shift_ms * FLAGS.sample_rate) / 1000)
  # Figure out the learning rates for each training phase. Since it's often
  # effective to have high learning rates at the start of training, followed by
  # lower levels towards the end, the number of steps and learning rates can be
  # specified as comma-separated lists to define the rate at each stage. For
  # example --how_many_training_steps=10000,3000 --learning_rate=0.001,0.0001
  # will run 13,000 training loops in total, with a rate of 0.001 for the first
  # 10,000, and 0.0001 for the final 3,000.
  training_steps_list = list(map(int, FLAGS.how_many_training_steps.split(',')))
  learning_rates_list = list(map(float, FLAGS.learning_rate.split(',')))
  if len(training_steps_list) != len(learning_rates_list):
    raise Exception(
        '--how_many_training_steps and --learning_rate must be equal length '
        'lists, but are %d and %d long instead' % (len(training_steps_list),
                                                   len(learning_rates_list)))

  actual_batch_size = tf.placeholder(tf.int32, [1])

  fingerprint_input = tf.placeholder(
      tf.float32, [None, fingerprint_size], name='fingerprint_input')

  hidden, logits, dropout_prob = models.create_model(
      fingerprint_input,
      model_settings,
      FLAGS.model_architecture,
      is_training=True)

  # Define loss and optimizer
  ground_truth_input = tf.placeholder(
      tf.int64, [None], name='groundtruth_input')

  # Optionally we can add runtime checks to spot when NaNs or other symptoms of
  # numerical errors start occurring during training.
  control_dependencies = []
  if FLAGS.check_nans:
    checks = tf.add_check_numerics_ops()
    control_dependencies = [checks]

  # Create the back propagation and training evaluation machinery in the graph.
  with tf.name_scope('cross_entropy'):
    cross_entropy_mean = tf.losses.sparse_softmax_cross_entropy(
        labels=tf.slice(ground_truth_input,[0],actual_batch_size),
        logits=tf.slice(logits,[0,0],tf.concat([actual_batch_size,[-1]],0)))
  tf.summary.scalar('cross_entropy', cross_entropy_mean)
  with tf.name_scope('train'), tf.control_dependencies(control_dependencies):
    learning_rate_input = tf.placeholder(
        tf.float32, [], name='learning_rate_input')
    if FLAGS.optimizer=='sgd':
      train_step = tf.train.GradientDescentOptimizer(
            learning_rate_input).minimize(cross_entropy_mean)
    elif FLAGS.optimizer=='adam':
      train_step = tf.train.AdamOptimizer(
            learning_rate_input).minimize(cross_entropy_mean)
    elif FLAGS.optimizer=='adagrad':
      train_step = tf.train.AdagradOptimizer(
            learning_rate_input).minimize(cross_entropy_mean)
    elif FLAGS.optimizer=='rmsprop':
      train_step = tf.train.RMSPropOptimizer(
            learning_rate_input).minimize(cross_entropy_mean)
  predicted_indices = tf.argmax(logits, 1)
  correct_prediction = tf.equal(predicted_indices, ground_truth_input)
  confusion_matrix = tf.confusion_matrix(
      tf.slice(ground_truth_input,[0],actual_batch_size),
      tf.slice(predicted_indices,[0],actual_batch_size),
      num_classes=label_count)
  evaluation_step = tf.reduce_mean(tf.cast(tf.slice(
                                   correct_prediction,[0],actual_batch_size), tf.float32))
  tf.summary.scalar('accuracy', evaluation_step)

  global_step = tf.train.get_or_create_global_step()
  increment_global_step = tf.assign(global_step, global_step + 1)

  saver = tf.train.Saver(tf.global_variables(), max_to_keep=0)

  # Merge all the summaries and write them out to /tmp/retrain_logs (by default)
  merged_summaries = tf.summary.merge_all()
  train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/train',
                                       sess.graph)
  validation_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/validation')

  tf.global_variables_initializer().run()

  start_step = 1

  if FLAGS.start_checkpoint:
    models.load_variables_from_checkpoint(sess, FLAGS.start_checkpoint)
    start_step = 1 + global_step.eval(session=sess)

  t0 = dt.datetime.now()
  tf.logging.info('Training from time %s, step: %d ', t0.isoformat(), start_step)

  # Save graph.pbtxt.
  tf.train.write_graph(sess.graph_def, FLAGS.train_dir,
                       FLAGS.model_architecture + '.pbtxt')

  # Save list of words.
  if FLAGS.start_checkpoint=='':
    with gfile.GFile(os.path.join(FLAGS.train_dir, \
                                  FLAGS.model_architecture + '_labels.txt'), 'w') as f:
      f.write(FLAGS.wanted_words.replace(',','\n'))

  # log complexity of model
  total_parameters = 0
  for variable in tf.trainable_variables():
      shape = variable.get_shape()
      variable_parameters = 1
      for dim in shape:
          variable_parameters *= int(dim)
      total_parameters += variable_parameters
  tf.logging.info('number of trainable parameters: %d',total_parameters)

  checkpoint_path = os.path.join(FLAGS.train_dir,
                                 FLAGS.model_architecture + '.ckpt')
  if FLAGS.start_checkpoint=='':
    tf.logging.info('Saving to "%s-%d"', checkpoint_path, 0)
    saver.save(sess, checkpoint_path, global_step=0)

  # exit if how_many_training_steps==0
  if FLAGS.how_many_training_steps=='0':
      return

  training_set_size = audio_processor.set_size('training')
  testing_set_size = audio_processor.set_size('testing')
  validation_set_size = audio_processor.set_size('validation')

  # Training loop.
  training_steps_max = np.sum(training_steps_list)
  for training_step in xrange(start_step, training_steps_max + 1):
    if training_set_size>0 and FLAGS.save_step_interval>0:
      # Figure out what the current learning rate is.
      training_steps_sum = 0
      for i in range(len(training_steps_list)):
        training_steps_sum += training_steps_list[i]
        if training_step <= training_steps_sum:
          learning_rate_value = learning_rates_list[i]
          break
      # Pull the audio samples we'll use for training.
      train_fingerprints, train_ground_truth, _ = audio_processor.get_data(
          FLAGS.batch_size, 0, model_settings, FLAGS.background_frequency,
          FLAGS.background_volume, time_shift_samples, FLAGS.time_shift_random, 'training', sess)
      # Run the graph with this batch of training data.
      train_summary, train_accuracy, cross_entropy_value, _, _ = sess.run(
          [
              merged_summaries, evaluation_step, cross_entropy_mean, train_step,
              increment_global_step
          ],
          feed_dict={
              fingerprint_input: train_fingerprints,
              ground_truth_input: train_ground_truth,
              learning_rate_input: learning_rate_value,
              actual_batch_size: [FLAGS.batch_size],
              dropout_prob: model_settings['dropout_prob']
          })
      train_writer.add_summary(train_summary, training_step)
      t1=dt.datetime.now()-t0
      tf.logging.info('Elapsed %f, Step #%d: rate %f, accuracy %.1f%%, cross entropy %f' %
                      (t1.total_seconds(), training_step, learning_rate_value, train_accuracy * 100,
                       cross_entropy_value))

      # Save the model checkpoint periodically.
      if (training_step % FLAGS.save_step_interval == 0 or
          training_step == training_steps_max):
        tf.logging.info('Saving to "%s-%d"', checkpoint_path, training_step)
        saver.save(sess, checkpoint_path, global_step=training_step)

    is_last_step = (training_step == training_steps_max)
    if validation_set_size>0 and (is_last_step or (training_step % FLAGS.eval_step_interval) == 0):
      validate_and_test('validation', validation_set_size, model_settings, \
                        time_shift_samples, sess, merged_summaries, evaluation_step, \
                        confusion_matrix, logits, hidden, validation_writer, \
                        audio_processor, is_last_step, fingerprint_input, \
                        ground_truth_input, actual_batch_size, dropout_prob, \
                        training_step, t0)
  if testing_set_size>0:
    validate_and_test('testing', testing_set_size, model_settings, time_shift_samples, \
                      sess, merged_summaries, evaluation_step, confusion_matrix, \
                      logits, hidden, validation_writer, audio_processor, \
                      True, fingerprint_input, ground_truth_input, \
                      actual_batch_size, dropout_prob, training_steps_max, t0)

def validate_and_test(set_kind, set_size, model_settings, time_shift_samples, sess, \
                      merged_summaries, evaluation_step, confusion_matrix, logits, \
                      hidden, validation_writer, audio_processor, is_last_step, \
                      fingerprint_input, ground_truth_input, actual_batch_size, \
                      dropout_prob, training_step, t0):
  total_accuracy = 0
  total_conf_matrix = None
  for isample in xrange(0, set_size, FLAGS.batch_size):
    fingerprints, ground_truth, samples = (
        audio_processor.get_data(FLAGS.batch_size, isample, model_settings, 0.0, 0.0,
                                 0.0 if FLAGS.time_shift_random else time_shift_samples,
                                 FLAGS.time_shift_random,
                                 set_kind, sess))
    needed = FLAGS.batch_size - fingerprints.shape[0]
    if needed>0:
      fingerprints = np.append(fingerprints,
            np.repeat(fingerprints[[0],:],needed,axis=0), axis=0)
      ground_truth = np.append(ground_truth,
            np.repeat(ground_truth[[0]],needed,axis=0), axis=0)
      for _ in range(needed):
        samples.append(samples[0])
    # Run a validation step and capture training summaries for TensorBoard
    # with the `merged` op.
    summary, accuracy, conf_matrix, logit_vals, hidden_vals = sess.run(
        [merged_summaries, evaluation_step, confusion_matrix, logits, hidden],
        feed_dict={
            fingerprint_input: fingerprints,
            ground_truth_input: ground_truth,
            actual_batch_size: [FLAGS.batch_size - needed],
            dropout_prob: 1.0
        })
    if set_kind=='validation':
      validation_writer.add_summary(summary, training_step)
    batch_size = min(FLAGS.batch_size, set_size - isample)
    total_accuracy += (accuracy * batch_size) / set_size
    if total_conf_matrix is None:
      total_conf_matrix = conf_matrix
    else:
      total_conf_matrix += conf_matrix
    obtained = FLAGS.batch_size - needed
    if isample==0:
      samples_data = [None]*set_size
      groundtruth_data = np.empty((set_size,))
      logit_data = np.empty((set_size,np.shape(logit_vals)[1]))
    samples_data[isample:isample+obtained] = samples[:obtained]
    groundtruth_data[isample:isample+obtained] = ground_truth[:obtained]
    logit_data[isample:isample+obtained,:] = logit_vals[:obtained,:]
    if is_last_step:
      if FLAGS.save_hidden:
        if isample==0:
          hidden_layers = []
          for ihidden in range(len(hidden_vals)):
            nHWC = np.shape(hidden_vals[ihidden])[1:]
            hidden_layers.append(np.empty((set_size, *nHWC)))
        for ihidden in range(len(hidden_vals)):
          hidden_layers[ihidden][isample:isample+obtained,:,:] = \
                hidden_vals[ihidden][:obtained,:,:,:]
      if FLAGS.save_fingerprints:
        if isample==0:
          nW = round((FLAGS.clip_duration_ms - FLAGS.window_size_ms) / \
                     FLAGS.window_stride_ms + 1)
          nH = round(np.shape(fingerprints)[1]/nW)
          input_layer = np.empty((set_size,nW,nH))
        input_layer[isample:isample+obtained,:,:] = \
              np.reshape(fingerprints[:obtained,:],(obtained,nW,nH))
  tf.logging.info('Confusion Matrix:\n %s\n %s' % \
                  (audio_processor.words_list,total_conf_matrix))
  t1=dt.datetime.now()-t0
  tf.logging.info('Elapsed %f, Step %d: %s accuracy = %.1f%% (N=%d)' %
                  (t1.total_seconds(), training_step, set_kind.capitalize(), \
                   total_accuracy * 100, set_size))
  np.savez(os.path.join(FLAGS.train_dir, \
           'logits.'+set_kind+'.ckpt-'+str(training_step)+'.npz'), \
           samples=samples_data, groundtruth=groundtruth_data, logits=logit_data)
  if is_last_step:
    if FLAGS.save_hidden:
      np.savez(os.path.join(FLAGS.data_dir,'hidden.npz'), \
               *hidden_layers, samples=samples_data)
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
      '--background_volume',
      type=float,
      default=0.1,
      help="""\
      How loud the background noise should be, between 0 and 1.
      """)
  parser.add_argument(
      '--background_frequency',
      type=float,
      default=0.8,
      help="""\
      How many of the training samples have background noise mixed in.
      """)
  parser.add_argument(
      '--silence_percentage',
      type=float,
      default=10.0,
      help="""\
      How much of the training data should be silence.
      """)
  parser.add_argument(
      '--unknown_percentage',
      type=float,
      default=10.0,
      help="""\
      How much of the training data should be unknown words.
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
      type=int,
      default=1,
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
      '--how_many_training_steps',
      type=str,
      default='15000,3000',
      help='How many training loops to run',)
  parser.add_argument(
      '--eval_step_interval',
      type=int,
      default=400,
      help='How often to evaluate the training results.')
  parser.add_argument(
      '--learning_rate',
      type=str,
      default='0.001,0.0001',
      help='How large a learning rate to use when training.')
  parser.add_argument(
      '--batch_size',
      type=int,
      default=100,
      help='How many items to train with at once',)
  parser.add_argument(
      '--summaries_dir',
      type=str,
      default='/tmp/retrain_logs',
      help='Where to save summary logs for TensorBoard.')
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
      '--train_dir',
      type=str,
      default='/tmp/speech_commands_train',
      help='Directory to write event logs and checkpoint.')
  parser.add_argument(
      '--save_step_interval',
      type=int,
      default=100,
      help='Save model checkpoint every save_steps.')
  parser.add_argument(
      '--start_checkpoint',
      type=str,
      default='',
      help='If specified, restore this pretrained model before any training.')
  parser.add_argument(
      '--filter_counts',
      type=str,
      default='64,64,64',
      help='A vector of length 3 specifying how many filters to use for the conv layers in the conv and vgg models')
  parser.add_argument(
      '--filter_sizes',
      type=str,
      default='3,3,3',
      help='A vector of length 3 specifying the filter sizes to use for the conv layers in the vgg model')
  parser.add_argument(
      '--final_filter_len',
      type=int,
      default=[110],
      help='The length of the final conv1d layer in the vgg model.  Must be even.')
  parser.add_argument(
      '--random_seed_batch',
      type=int,
      default=59185,
      help='Randomize mini-batch selection if -1; otherwise use supplied number as seed.')
  parser.add_argument(
      '--random_seed_weights',
      type=int,
      default=59185,
      help='Randomize weight initialization if -1; otherwise use supplied number as seed.')
  parser.add_argument(
      '--dilate_after_layer',
      type=int,
      default=65535,
      help='Convolutional layer at which to start exponentially dilating.')
  parser.add_argument(
      '--stride_after_layer',
      type=int,
      default=65535,
      help='Convolutional layer at which to start striding by 2.')
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
      '--dropout_prob',
      type=float,
      default=0.5,
      help='Dropout probability during training')
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
      '--connection_type',
      type=str,
      default='plain',
      help='Either plain or residual.')
  parser.add_argument(
      '--check_nans',
      type=str2bool,
      default=False,
      help='Whether to check for invalid numbers during processing')
  parser.add_argument(
      '--save_hidden',
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
