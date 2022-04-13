#!/usr/bin/python3

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
`--labels_touse` argument. In this case, 'up,down' might be what you want, and
the audio in the 'other' folder would be used to train an 'unknown' category.

To pull this all together, you'd run:

bazel run tensorflow/examples/speech_commands:train -- \
--data_dir=my_wavs --labels_touse=up,down

"""
import argparse
import os.path
import sys

import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import *

import data
import models

from subprocess import run, PIPE, STDOUT
from datetime import datetime
import socket

import json

import importlib

FLAGS = None

# Create the back propagation and training evaluation machinery
def loss_fn(logits, ground_truth_input):
    return tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=ground_truth_input,
            logits=logits[:,0,:])

def evaluate_fn(model, fingerprint_input, ground_truth_input, istraining):
    hidden_activations, logits = model(fingerprint_input, training=istraining)
    predicted_indices = tf.math.argmax(logits, 2)[:,0]
    correct_prediction = tf.equal(predicted_indices, ground_truth_input)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return logits, accuracy, predicted_indices, hidden_activations

def main():
  os.environ['TF_DETERMINISTIC_OPS']=FLAGS.deterministic
  os.environ['TF_DISABLE_SPARSE_SOFTMAX_XENT_WITH_LOGITS_OP_DETERMINISM_EXCEPTIONS']=FLAGS.deterministic

  sys.path.append(os.path.dirname(FLAGS.model_architecture))
  model = importlib.import_module(os.path.basename(FLAGS.model_architecture))

  flags = vars(FLAGS)
  for key in sorted(flags.keys()):
    print('%s = %s' % (key, flags[key]))

  FLAGS.model_parameters = json.loads(FLAGS.model_parameters)

  if FLAGS.random_seed_weights!=-1:
      tf.random.set_seed(FLAGS.random_seed_weights)

  for physical_device in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(physical_device, True)
  tf.config.set_soft_device_placement(True)

  if FLAGS.start_checkpoint:
    labels_touse = tf.io.gfile.GFile(os.path.join(FLAGS.train_dir, 'labels.txt')).readlines()
    labels_touse = ','.join([x.rstrip() for x in labels_touse])
    if labels_touse != FLAGS.labels_touse:
      if set(labels_touse.split(',')) != set(FLAGS.labels_touse.split(',')):
        print('ERROR: labels_touse does not match model being restored from')
        return
      else:
        print('WARNING: labels_touse is out of order.  continuing with order from restored model')
  else:
    labels_touse = FLAGS.labels_touse

  nlabels = len(labels_touse.split(','))

  model_settings = models.prepare_model_settings(
      nlabels,
      FLAGS.audio_tic_rate,
      FLAGS.nchannels,
      1,
      FLAGS.batch_size,
      FLAGS.context_ms,
      FLAGS.model_parameters)

  audio_processor = data.AudioProcessor(
      FLAGS.data_dir,
      FLAGS.shiftby_ms,
      labels_touse.split(','), FLAGS.kinds_touse.split(','),
      FLAGS.validation_percentage, FLAGS.validation_offset_percentage,
      FLAGS.validation_files.split(','),
      0, FLAGS.testing_files.split(','), FLAGS.subsample_skip,
      FLAGS.subsample_label,
      FLAGS.partition_label, FLAGS.partition_n, FLAGS.partition_training_files.split(','),
      FLAGS.partition_validation_files.split(','),
      FLAGS.random_seed_batch,
      FLAGS.testing_equalize_ratio, FLAGS.testing_max_sounds,
      model_settings,
      FLAGS.data_loader_queuesize, FLAGS.data_loader_maxprocs)

  thismodel = model.create_model(model_settings)
  thismodel.summary()

  thisoptimizer = getattr(tf.keras.optimizers, FLAGS.optimizer)(learning_rate=FLAGS.learning_rate)

  start_step = 1

  checkpoint = tf.train.Checkpoint(thismodel=thismodel, thisoptimizer=thisoptimizer)
  checkpoint_basepath = os.path.join(FLAGS.train_dir, 'ckpt')

  if FLAGS.start_checkpoint:
    checkpoint.read(FLAGS.start_checkpoint)
    start_step = 1 + int(os.path.basename(FLAGS.start_checkpoint).split('-')[-1])
  else:
    print('Saving to "%s-%d"' % (checkpoint_basepath, 0))
    checkpoint.write(checkpoint_basepath+'-0')

  t0 = datetime.now()
  print('Training from time %s, step: %d ' % (t0.isoformat(), start_step))

  if not FLAGS.start_checkpoint:
    with tf.io.gfile.GFile(os.path.join(FLAGS.train_dir, 'labels.txt'), 'w') as f:
      f.write(labels_touse.replace(',','\n'))

  train_writer = tf.summary.create_file_writer(FLAGS.summaries_dir + '/train')
  validation_writer = tf.summary.create_file_writer(FLAGS.summaries_dir + '/validation')

  # exit if how_many_training_steps==0
  if FLAGS.how_many_training_steps==0:
      # pre-process a batch of data to make sure settings are valid
      train_fingerprints, train_ground_truth, _ = audio_processor.get_data(
          FLAGS.batch_size, 0, model_settings, FLAGS.shiftby_ms, 'training')
      evaluate_fn(thismodel, train_fingerprints, train_ground_truth, False)
      return

  training_set_size = audio_processor.set_size('training')
  testing_set_size = audio_processor.set_size('testing')
  validation_set_size = audio_processor.set_size('validation')

  @tf.function
  def train_step(train_fingerprints, train_ground_truth):
    # Run the graph with this batch of training data.
    with tf.GradientTape() as tape:
      logits, train_accuracy, _, _ = evaluate_fn(thismodel, train_fingerprints,
                                                 train_ground_truth, True)
      loss_value = loss_fn(logits, train_ground_truth)
    gradients = tape.gradient(loss_value, thismodel.trainable_variables)
    thisoptimizer.apply_gradients(zip(gradients, thismodel.trainable_variables))
    cross_entropy_mean = tf.math.reduce_mean(loss_value)
    return cross_entropy_mean, train_accuracy

  # Training loop.
  print('line format is Elapsed %f, Step #%d, Train accuracy %.1f, cross entropy %f')
  for training_step in range(start_step, FLAGS.how_many_training_steps + 1):
    if training_set_size>0:
      # Pull the sounds we'll use for training.
      train_fingerprints, train_ground_truth, _ = audio_processor.get_data(
          FLAGS.batch_size, 0, model_settings, FLAGS.shiftby_ms, 'training')

      cross_entropy_mean, train_accuracy = train_step(tf.constant(train_fingerprints),
                                                      tf.constant(train_ground_truth))
      t1=datetime.now()-t0

      with train_writer.as_default():
        tf.summary.scalar('cross_entropy', cross_entropy_mean, step=training_step)
        tf.summary.scalar('accuracy', train_accuracy, step=training_step)

      print('%f,%d,%.1f,%f' %
                      (t1.total_seconds(), training_step,
                       train_accuracy.numpy() * 100, cross_entropy_mean.numpy()))

      # Save the model checkpoint periodically.
      if ((FLAGS.save_step_period > 0 and training_step % FLAGS.save_step_period == 0) or
          training_step == FLAGS.how_many_training_steps):
        print('Saving to "%s-%d"' % (checkpoint_basepath, training_step))
        checkpoint.write(checkpoint_basepath+'-'+str(training_step))

    if validation_set_size>0 and \
       training_step != FLAGS.how_many_training_steps and \
       (FLAGS.validate_step_period > 0 and training_step % FLAGS.validate_step_period == 0):
      validate_and_test(thismodel, 'validation', validation_set_size, model_settings, \
                        audio_processor, False, training_step, t0, nlabels, \
                        validation_writer)

  if validation_set_size>0:
    validate_and_test(thismodel, 'validation', validation_set_size, model_settings, \
                      audio_processor, True, FLAGS.how_many_training_steps, t0, nlabels, \
                      validation_writer)
  if testing_set_size>0:
    validate_and_test(thismodel, 'testing', testing_set_size, model_settings, \
                      audio_processor, True, FLAGS.how_many_training_steps, t0, nlabels, \
                      validation_writer)

@tf.function
def validate_test_step(model, nlabels, fingerprints, ground_truth):
  needed = FLAGS.batch_size - fingerprints.shape[0]
  logits, accuracy, predicted_indices, hidden_activations = evaluate_fn(model, fingerprints,
                                                                        ground_truth, False)
  loss_value = loss_fn(logits, ground_truth)
  cross_entropy_mean = tf.math.reduce_mean(loss_value)
  confusion_matrix = tf.math.confusion_matrix(ground_truth,
                                              predicted_indices,
                                              num_classes=nlabels)

  return needed, logits, accuracy, hidden_activations, cross_entropy_mean, confusion_matrix

def validate_and_test(model, set_kind, set_size, model_settings, \
                      audio_processor, is_last_step, training_step, t0, nlabels, \
                      validation_writer):
  total_accuracy = 0
  total_conf_matrix = None
  for isound in range(0, set_size, FLAGS.batch_size):
    fingerprints, ground_truth, sounds = (
        audio_processor.get_data(FLAGS.batch_size, isound, model_settings,
                                 FLAGS.shiftby_ms, set_kind))
    needed, logits, accuracy, hidden_activations, cross_entropy_mean, confusion_matrix = \
            validate_test_step(model, tf.constant(nlabels),
                               tf.constant(fingerprints), tf.constant(ground_truth))

    with validation_writer.as_default():
      tf.summary.scalar('cross_entropy', cross_entropy_mean, step=training_step)
      tf.summary.scalar('accuracy', accuracy, step=training_step)

    batch_size = min(FLAGS.batch_size, set_size - isound)
    total_accuracy += (accuracy * batch_size) / set_size
    if total_conf_matrix is None:
      total_conf_matrix = confusion_matrix
    else:
      total_conf_matrix += confusion_matrix
    obtained = FLAGS.batch_size - needed
    if isound==0:
      sounds_data = [None]*set_size
      groundtruth_data = np.empty((set_size,))
      logit_data = np.empty((set_size,np.shape(logits)[2]))
    sounds_data[isound:isound+obtained] = sounds
    groundtruth_data[isound:isound+obtained] = ground_truth
    logit_data[isound:isound+obtained,:] = logits[:,0,:]
    if is_last_step:
      if FLAGS.save_hidden:
        if isound==0:
          hidden_layers = []
          for ihidden in range(len(hidden_activations)):
            nHWC = np.shape(hidden_activations[ihidden])[1:]
            hidden_layers.append(np.empty((set_size, *nHWC)))
        for ihidden in range(len(hidden_activations)):
          hidden_layers[ihidden][isound:isound+obtained,...] = hidden_activations[ihidden]
      if FLAGS.save_fingerprints:
        if isound==0:
          nHWC = np.shape(fingerprints)[1:]
          input_layer = np.empty((set_size ,*nHWC))
        input_layer[isound:isound+obtained,...] = fingerprints
  print('Confusion Matrix:\n %s\n %s' % \
                  (audio_processor.labels_list, total_conf_matrix.numpy()))
  t1=datetime.now()-t0
  print('%f,%d,%.1f %s' %
                  (t1.total_seconds(), training_step, \
                   total_accuracy * 100, set_kind.capitalize()))
  np.savez(os.path.join(FLAGS.train_dir,
                        'logits.'+set_kind+'.ckpt-'+str(training_step)+'.npz'), \
           sounds=sounds_data, groundtruth=groundtruth_data, logits=logit_data)
  if is_last_step:
    if FLAGS.save_hidden:
      np.savez(os.path.join(FLAGS.train_dir,
                            'hidden.'+set_kind+'.ckpt-'+str(training_step)+'.npz'), \
               *hidden_layers, sounds=sounds_data)
    if FLAGS.save_fingerprints:
      np.savez(os.path.join(FLAGS.train_dir,
                            'fingerprints.'+set_kind+'.ckpt-'+str(training_step)+'.npz'), \
              input_layer, sounds=sounds_data)

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
      '--data_dir',
      type=str,
      default='/tmp/speech_dataset/',
      help="""\
      Where to download the speech training data to.
      """)
  parser.add_argument(
      '--shiftby_ms',
      type=float,
      default=100.0,
      help="""\
      Range to shift the training audio by in time.
      """)
  parser.add_argument(
      '--testing_files',
      type=str,
      default='',
      help='Which wav files to use as a test set.')
  parser.add_argument(
      '--subsample_label',
      type=str,
      default='',
      help='Train on only a subset of annotations for this label.')
  parser.add_argument(
      '--subsample_skip',
      type=str,
      default='',
      help='Take only every Nth annotation for the specified label.')
  parser.add_argument(
      '--partition_label',
      type=str,
      default='',
      help='Train on only a fixed number of annotations for this label.')
  parser.add_argument(
      '--partition_n',
      type=int,
      default=0,
      help='Train on only this number of annotations from each file for the specified label.')
  parser.add_argument(
      '--partition_training_files',
      type=str,
      default='',
      help='Train on only these files for the specified label.')
  parser.add_argument(
      '--partition_validation_files',
      type=str,
      default='',
      help='Validate on only these files for the specified label.')
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
      '--learning_rate',
      type=float,
      default=0.001,
      help='How large a learning rate to use when training.')
  parser.add_argument(
      '--optimizer',
      type=str,
      default='sgd',
      help='What optimizer to use.  One of Adadelta, Adagrad, Adam, Adamax, Ftrl, Nadam, RMSProp, or SGD.')
  parser.add_argument(
      '--how_many_training_steps',
      type=int,
      default=15000,
      help='How many training loops to run',)
  parser.add_argument(
      '--validate_step_period',
      type=int,
      default=400,
      help='How often to evaluate the training results.')
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
      '--labels_touse',
      type=str,
      default='yes,no,up,down,left,right,on,off,stop,go',
      help='Words to use (others will be added to an unknown label)',)
  parser.add_argument(
      '--kinds_touse',
      type=str,
      default='annotated,classified',
      help='A comma-separted list of "annotated", "detected" , or "classified"',)
  parser.add_argument(
      '--train_dir',
      type=str,
      default='/tmp/speech_commands_train',
      help='Directory to write event logs and checkpoint.')
  parser.add_argument(
      '--save_step_period',
      type=int,
      default=100,
      help='Save model checkpoint every save_steps.')
  parser.add_argument(
      '--start_checkpoint',
      type=str,
      default='',
      help='If specified, restore this pretrained model before any training.')
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
      '--testing_equalize_ratio',
      type=int,
      default=0,
      help='Limit most common label to be no more than this times more than the least common label for testing.')
  parser.add_argument(
      '--testing_max_sounds',
      type=int,
      default=0,
      help='Limit number of test sounds to this number.')
  parser.add_argument(
      '--model_architecture',
      type=str,
      default='convolutional',
      help='What model architecture to use')
  parser.add_argument(
      '--data_loader_queuesize',
      type=int,
      default=0,
      help='How many mini-batches to load in advance')
  parser.add_argument(
      '--data_loader_maxprocs',
      type=int,
      default=0,
      help='The limit of how many extra processes to use to load mini-batches')
  parser.add_argument(
      '--model_parameters',
      type=str,
      default='{}',
      help='What model parameters to use')
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
  parser.add_argument(
      '--deterministic',
      type=str,
      default='0',
      help='')

  FLAGS, unparsed = parser.parse_known_args()

  print(str(datetime.now())+": start time")
  repodir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
  with open(os.path.join(repodir, "VERSION.txt"), 'r') as fid:
    print('SongExplorer version = '+fid.read().strip().replace('\n',', '))
  print("hostname = "+socket.gethostname())
  print("CUDA_VISIBLE_DEVICES = "+os.environ.get('CUDA_VISIBLE_DEVICES',''))
  p = run('which nvidia-smi && nvidia-smi', shell=True, stdout=PIPE, stderr=STDOUT)
  print(p.stdout.decode('ascii').rstrip())

  try:
    main()

  except Exception as e:
    print(e)

  finally:
    os.sync()
    print(str(datetime.now())+": finish time")
