#!/usr/bin/env python3

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
#      --video_findfile=same-basename \
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

# use agg here as otherwise pims tries to open gtk
# see https://github.com/soft-matter/pims/issues/351
import matplotlib as mpl
mpl.use('Agg')
import pims

import importlib

import tifffile

FLAGS = None

def main():
  os.environ['TF_DETERMINISTIC_OPS']=FLAGS.deterministic
  os.environ['TF_DISABLE_SPARSE_SOFTMAX_XENT_WITH_LOGITS_OP_DETERMINISM_EXCEPTIONS']=FLAGS.deterministic

  sys.path.append(os.path.dirname(FLAGS.video_findfile))
  video_findfile = importlib.import_module(os.path.basename(FLAGS.video_findfile)).video_findfile

  audio_tic_rate = FLAGS.audio_tic_rate

  print('model = '+str(FLAGS.model))
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
  use_audio = thismodel.get_use_audio()
  use_video = thismodel.get_use_video()

  #Load a wav file and return audio_tic_rate and numpy data of float64 type.
  if use_audio:
    audio_data, fs = tf.audio.decode_wav(tf.io.read_file(FLAGS.wav))
    if fs.numpy() != audio_tic_rate:
      print("ERROR: audio_tic_rate of WAV file is %d while %d was expected" %
            (fs.numpy(), audio_tic_rate))
      exit()
    if audio_data.shape[1] != FLAGS.audio_nchannels:
      print("ERROR: audio_nchannels of WAV file is %d while %d was expected" %
            (audio_data.shape[1], FLAGS.audio_nchannels))
      exit()

  if use_video:
    sound_basename = os.path.basename(FLAGS.wav)
    sound_dirname =  os.path.dirname(FLAGS.wav)
    vidfile = video_findfile(sound_dirname, sound_basename)
    tiffile = os.path.join(sound_dirname, os.path.splitext(vidfile)[0]+".tif")
    bkg = tifffile.imread(tiffile)
    video_data = pims.open(os.path.join(sound_dirname, vidfile))
    video_channels = tf.cast([int(x) for x in FLAGS.video_channels.split(',')], tf.int32)
    video_frame_rate = FLAGS.video_frame_rate
    if video_data.frame_rate != video_frame_rate:
      print('ERROR: frame_rate of video file is %d when %d is expected' % (video_data.frame_rate, video_frame_rate))
      exit()
    if video_data.frame_shape[0] != FLAGS.video_frame_width:
      print('ERROR: frame_width of video file is %d when %d is expected' % (video_data.frame_shape[1], FLAGS.video_frame_width))
      exit()
    if video_data.frame_shape[1] != FLAGS.video_frame_height:
      print('ERROR: frame_height of video file is %d when %d is expected' % (video_data.frame_shape[1], FLAGS.video_frame_height))
      exit()
    if video_data.frame_shape[2] < max(video_channels):
      print('ERROR: nchannels of video file is %d when channel(s) %s is/are expected' % (video_data.frame_shape[2], FLAGS.video_channels))
      exit()

  if FLAGS.parallelize==1:
    print("WARNING: classify_parallelize in configuration.pysh is set to 1.  making predictions is faster if it is > 1")

  input_shape = thismodel.get_input_shape()
  if use_audio:
    clip_window_samples = input_shape[0][1].numpy() if use_video else input_shape[1].numpy()
    data_len_samples = audio_data.shape[0]
    data_sample_rate = audio_tic_rate
  elif use_video:
    clip_window_samples = input_shape[1].numpy()
    data_len_samples = len(video_data)
    data_sample_rate = video_frame_rate

  if use_video:
    clip_window_frames = input_shape[1][1].numpy() if use_audio else input_shape[1].numpy()
    video_slice = np.zeros((clip_window_frames,
                            FLAGS.video_frame_height,
                            FLAGS.video_frame_width,
                            len(video_channels)),
                           dtype=np.float32)

  context_samples = int(FLAGS.context_ms * data_sample_rate / 1000)
  stride_x_downsample_samples = (clip_window_samples - context_samples) // (FLAGS.parallelize-1)
  clip_stride_samples = stride_x_downsample_samples * FLAGS.parallelize

  stride_x_downsample_ms = stride_x_downsample_samples / data_sample_rate * 1000
  npadding = round((FLAGS.context_ms/2 + FLAGS.shiftby_ms) / stride_x_downsample_ms)
  probability_list = [np.zeros((npadding, len(labels)), dtype=np.float32)]

  # Inference along audio stream.
  for data_offset_samples in range(0, 1+data_len_samples, clip_stride_samples):
    start_sample = data_offset_samples
    stop_sample = data_offset_samples + clip_window_samples
    pad_len = stop_sample - data_len_samples
    
    if use_audio:
      audio_slice = audio_data[start_sample:stop_sample,:] if pad_len<=0 else \
                    np.pad(audio_data[start_sample:stop_sample,:],
                           ((0,pad_len),(0,0)), mode='median')
    if use_video:
      if use_audio:
        start_frame = round(start_sample / audio_tic_rate * video_frame_rate)
        stop_frame = start_frame + clip_window_frames
      else:
        start_frame, stop_frame = start_sample, stop_sample
      for iframe in range(start_frame, stop_frame):
        if iframe < len(video_data):
          video_slice[iframe-start_frame,:,:,:] = video_data[iframe][:,:,video_channels] - bkg[:,:,video_channels]
        else:
          video_slice[iframe-start_frame,:,:,:] = bkg[:,:,video_channels]

    if use_audio and use_video:
      inputs = [tf.expand_dims(audio_slice, 0), tf.expand_dims(video_slice, 0)]
    elif use_audio:
      inputs = tf.expand_dims(audio_slice, 0)
    elif use_video:
      inputs = tf.expand_dims(video_slice, 0)
    _,outputs = recognize_graph(inputs)

    current_time_ms = np.round(data_offset_samples * 1000 / data_sample_rate).astype(int)
    if pad_len>0:
      discard_len = np.ceil(pad_len/stride_x_downsample_samples).astype(int)
      probability_list.append(np.array(outputs.numpy()[0,:-discard_len,:]))
      break
    else:
      probability_list.append(np.array(outputs.numpy()[0,:,:]))

  sample_rate = round(1000/stride_x_downsample_ms)
  if sample_rate != 1000/stride_x_downsample_ms:
    print('WARNING: .wav files do not support fractional sampling rates!')

  probability_matrix = np.concatenate(probability_list)
  denominator = np.sum(probability_matrix * prevalences, axis=1)
  for ch in range(len(labels)):
    adjusted_probability = probability_matrix[:,ch] * prevalences[ch]
    adjusted_probability[npadding:] /= denominator[npadding:]
    #inotnan = ~isnan(probability_matrix(:,ch));
    waveform = adjusted_probability*np.iinfo(np.int16).max  # colon was inotnan
    filename = os.path.splitext(FLAGS.wav)[0]+'-'+labels[ch]+'.wav'
    wavfile.write(filename, int(sample_rate), waveform.astype('int16'))

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
      '--video_findfile',
      type=str,
      default='same-basename',
      help='What function to use to match WAV files to corresponding video files')
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
