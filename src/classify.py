#!/usr/bin/python3

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


# generate .wav files of per-label probabilities

# e.g.
# $SONGEXPLORER_BIN classify.sh \
#      --context_ms=204.8 \
#      --shiftby_ms=0.0 \
#      --model=`pwd`/trained-classifier/train_1k/frozen-graph.ckpt-50.pb \
#      --model_labels=`pwd`/trained-classifier/train_1k/labels.txt \
#      --wav=`pwd`/groundtruth-data/round1/20161207T102314_ch1_p1.wav \
#      --parallelize=65536 \
#      --labels=mel-pulse,mel-sine,ambient \
#      --prevalences=0.1,0.1,0.8


import argparse
import sys
import os

from datetime import datetime
import socket
from subprocess import run, PIPE, STDOUT

import numpy as np
import tensorflow as tf

from scipy.io import wavfile

FLAGS = None

def main():
  os.environ['TF_DETERMINISTIC_OPS']=FLAGS.deterministic

  #Load a wav file and return audio_tic_rate and numpy data of float64 type.
  data, audio_tic_rate = tf.audio.decode_wav(tf.io.read_file(FLAGS.wav))
  audio_tic_rate = audio_tic_rate.numpy()
  print('audio_tic_rate = '+str(audio_tic_rate))

  with open(FLAGS.model_labels, 'r') as fid:
    model_labels = fid.read().splitlines()

  if FLAGS.labels:
    labels = np.array(FLAGS.labels.split(','))
    prevalences = np.array([float(x) for x in FLAGS.prevalences.split(',')])
    if len(labels) != len(prevalences):
      print("ERROR: length of 'labels to use' (="+str(len(labels))+
            ") must equal length of 'prevalences' (="+str(len(prevalences))+")")
      exit()
    if set(labels) != set(model_labels):
      print("ERROR: 'labels to use' must be the same as "+
            os.path.join(logdir,model,'labels.txt'))
      exit()
    prevalences /= np.sum(prevalences)
    iimodel_labels = np.argsort(np.argsort(model_labels))
    ilabels = np.argsort(labels)
    labels = labels[ilabels][iimodel_labels]
    prevalences = prevalences[ilabels][iimodel_labels]
    assert np.all(labels==model_labels)
  else:
    labels = model_labels
    prevalences = [1/len(labels) for _ in range(len(labels))]
  print('labels: '+str(labels))
  print('prevalences: '+str(prevalences))

  # Load model and create a tf session to process audio pieces
  thismodel = tf.saved_model.load(FLAGS.model)
  recognize_graph = thismodel.inference_step

  clip_window_tics = thismodel.get_input_shape()[1].numpy()
  context_tics = np.round(FLAGS.context_ms * audio_tic_rate / 1000).astype(np.int)
  assert FLAGS.parallelize>1
  stride_x_downsample_tics = (clip_window_tics - context_tics) // (FLAGS.parallelize-1)
  clip_stride_tics = stride_x_downsample_tics * FLAGS.parallelize

  stride_x_downsample_ms = stride_x_downsample_tics/audio_tic_rate*1000
  npadding = int(round((FLAGS.context_ms/2+FLAGS.shiftby_ms)/stride_x_downsample_ms))
  probability_matrix = np.zeros((npadding, len(labels)))

  # Inference along audio stream.
  for audio_data_offset in range(0, 1+data.shape[0], clip_stride_tics):
    input_start = audio_data_offset
    input_end = audio_data_offset + clip_window_tics
    pad_len = input_end - data.shape[0]
    
    data_slice = tf.transpose(data[input_start:input_end,:] if pad_len<=0 else \
                              np.pad(data[input_start:input_end,:],
                                     ((0,pad_len),(0,0)), mode='median'))
    _,outputs = recognize_graph(tf.expand_dims(tf.transpose(data_slice), 0))
    current_time_ms = np.round(audio_data_offset * 1000 / audio_tic_rate).astype(np.int)
    if pad_len>0:
      discard_len = np.ceil(pad_len/stride_x_downsample_tics).astype(np.int)
      probability_matrix = np.concatenate((probability_matrix,
                                           np.array(outputs.numpy()[0,:-discard_len,:],
                                                    ndmin=2)))
      break
    else:
      probability_matrix = np.concatenate((probability_matrix,
                                           np.array(outputs.numpy()[0,:,:], ndmin=2)))

  tic_rate = round(1000/stride_x_downsample_ms)
  if tic_rate != 1000/stride_x_downsample_ms:
    print('WARNING: .wav files do not support fractional sampling rates!')

  denominator = np.sum(probability_matrix * prevalences, axis=1)
  for ch in range(len(labels)):
    adjusted_probability = probability_matrix[:,ch] * prevalences[ch]
    adjusted_probability[npadding:] /= denominator[npadding:]
    #inotnan = ~isnan(probability_matrix(:,ch));
    waveform = adjusted_probability*np.iinfo(np.int16).max  # colon was inotnan
    filename = os.path.splitext(FLAGS.wav)[0]+'-'+labels[ch]+'.wav'
    wavfile.write(filename, int(tic_rate), waveform.astype('int16'))

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='test_streaming_accuracy')
  parser.add_argument(
      '--wav', type=str, default='', help='The wave file path to evaluate.')
  parser.add_argument(
      '--model_labels',
      type=str,
      default='',
      help='The label file path containing all possible classes.')
  parser.add_argument(
      '--labels',
      type=str,
      default='',
      help='A comma-separated list of all possible classes.')
  parser.add_argument(
      '--prevalences',
      type=str,
      default='',
      help='A comma-separated list of the a priori probabilities of each label.')
  parser.add_argument(
      '--model', type=str, default='', help='The model used for inference')
  parser.add_argument(
      '--context_ms',
      type=float,
      default=1000,
      help='Length of each audio clip fed into model.')
  parser.add_argument(
      '--shiftby_ms',
      type=float,
      default=100.0,
      help="""\
      Range to shift the training audio by in time.
      """)
  parser.add_argument(
      '--parallelize',
      type=int,
      default=1,
      help='')
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
