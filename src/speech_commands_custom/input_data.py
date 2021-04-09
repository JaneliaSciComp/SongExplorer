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
import hashlib
import math
import os.path
import random
import re
import sys
import tarfile
import csv
import scipy.io.wavfile as spiowav

import numpy as np
import tensorflow as tf

from representation import *

MAX_NUM_WAVS_PER_CLASS = 2**27 - 1  # ~134M


def which_set(filename, validation_percentage, validation_offset_percentage, testing_percentage):
  """Determines which data partition the file should belong to.

  We want to keep files in the same training, validation, or testing sets even
  if new ones are added over time. This makes it less likely that testing
  samples will accidentally be reused in training when long runs are restarted
  for example. To keep this stability, a hash of the filename is taken and used
  to determine which set it should belong to. This determination only depends on
  the name and the set proportions, so it won't change as other files are added.

  It's also useful to associate particular files as related (for example words
  spoken by the same person), so anything after '_nohash_' in a filename is
  ignored for set determination. This ensures that 'bobby_nohash_0.wav' and
  'bobby_nohash_1.wav' are always in the same set, for example.

  Args:
    filename: File path of the data sample.
    validation_percentage: How much of the data set to use for validation.
    validation_offset_percentage: Which part of the data set to use for validation.
    testing_percentage: How much of the data set to use for testing.

  Returns:
    String, one of 'training', 'validation', or 'testing'.
  """
  base_name = os.path.basename(filename)
  # We want to ignore anything after '_nohash_' in the file name when
  # deciding which set to put a wav in, so the data set creator has a way of
  # grouping wavs that are close variations of each other.
  hash_name = re.sub(r'_nohash_.*$', '', base_name)
  # This looks a bit magical, but we need to decide whether this file should
  # go into the training, testing, or validation sets, and we want to keep
  # existing files in the same set even if more files are subsequently
  # added.
  # To do that, we need a stable way of deciding based on just the file name
  # itself, so we do a hash of that and then use that to generate a
  # probability value that we use to assign it.
  hash_name_hashed = hashlib.sha1(tf.compat.as_bytes(hash_name)).hexdigest()
  percentage_hash = ((int(hash_name_hashed, 16) %
                      (MAX_NUM_WAVS_PER_CLASS + 1)) *
                     (100.0 / MAX_NUM_WAVS_PER_CLASS))
  if percentage_hash < testing_percentage:
    result = 'testing'
  elif percentage_hash > (testing_percentage + validation_offset_percentage) and \
       percentage_hash < (testing_percentage + validation_offset_percentage + validation_percentage):
    result = 'validation'
  else:
    result = 'training'
  return result


class AudioProcessor(object):
  """Handles loading, partitioning, and preparing audio training data."""

  def __init__(self, data_dir,
               time_shift_ms, time_shift_random,
               wanted_words, labels_touse,
               validation_percentage, validation_offset_percentage, validation_files,
               testing_percentage, testing_files, subsample_skip, subsample_word,
               partition_word, partition_n, partition_training_files, partition_validation_files,
               random_seed_batch,
               testing_equalize_ratio, testing_max_samples, model_settings):
    self.data_dir = data_dir
    random.seed(None if random_seed_batch==-1 else random_seed_batch)
    np.random.seed(None if random_seed_batch==-1 else random_seed_batch)
    self.prepare_data_index(time_shift_ms, time_shift_random,
                            wanted_words, labels_touse,
                            validation_percentage, validation_offset_percentage, validation_files,
                            testing_percentage, testing_files, subsample_skip, subsample_word,
                            partition_word, partition_n, partition_training_files, partition_validation_files,
                            testing_equalize_ratio, testing_max_samples,
                            model_settings)
    self.prepare_processing_graph(model_settings)

  def prepare_data_index(self,
                         time_shift_ms, time_shift_random,
                         wanted_words, labels_touse,
                         validation_percentage, validation_offset_percentage, validation_files,
                         testing_percentage, testing_files, subsample_skip, subsample_word,
                         partition_word, partition_n, partition_training_files, partition_validation_files,
                         testing_equalize_ratio, testing_max_samples,
                         model_settings):
    """Prepares a list of the samples organized by set and label.

    The training loop needs a list of all the available data, organized by
    which partition it should belong to, and with ground truth labels attached.
    This function analyzes the folders below the `data_dir`, figures out the
    right
    labels for each file based on the name of the subdirectory it belongs to,
    and uses a stable hash to assign it to a data set partition.

    Args:
      wanted_words: Labels of the classes we want to be able to recognize.
      validation_percentage: How much of the data set to use for validation.
      validation_offset_percentage: Which part of the data set to use for validation.
      testing_percentage: How much of the data set to use for testing.

    Returns:
      Dictionary containing a list of file information for each set partition,
      and a lookup map for each class to determine its numeric index.

    Raises:
      Exception: If expected files are not found.
    """
    time_shift_samples = int((time_shift_ms * model_settings["sample_rate"]) / 1000)
    # Make sure the shuffling is deterministic.
    wanted_words_index = {}
    for index, wanted_word in enumerate(wanted_words):
      wanted_words_index[wanted_word] = index
    self.data_index = {'validation': [], 'testing': [], 'training': []}
    all_words = {}
    # Look through all the subfolders to find audio samples
    desired_samples = model_settings['desired_samples']
    search_path = os.path.join(self.data_dir, '*', '*.csv')
    wav_nsamples = {}
    subsample = {x:int(y) for x,y in zip(subsample_word.split(','),subsample_skip.split(','))
                          if x != ''}
    partition_words = partition_word.split(',')
    if '' in partition_words:
      partition_words.remove('')
    for csv_path in tf.io.gfile.glob(search_path):
      annotation_reader = csv.reader(open(csv_path))
      annotation_list = list(annotation_reader)
      if len(partition_words)>0:
        random.shuffle(annotation_list)
      for (iannotation, annotation) in enumerate(annotation_list):
        wavfile=annotation[0]
        ticks=[int(annotation[1]),int(annotation[2])]
        kind=annotation[3]
        word=annotation[4]
        if kind not in labels_touse:
          continue
        wav_path=os.path.join(os.path.dirname(csv_path),wavfile)
        if word in subsample and iannotation % subsample[word] != 0:
          continue
        if word in partition_words:
          if wavfile not in partition_training_files and \
             wavfile not in partition_validation_files:
            continue
          if wavfile in partition_training_files and \
             sum([x['label']==word and x['file']==wav_path \
                  for x in self.data_index['training']]) >= partition_n:
            continue
        if wav_path not in wav_nsamples:
          _, data = spiowav.read(wav_path, mmap=True)
          wav_nsamples[wav_path] = len(data)
        nsamples = wav_nsamples[wav_path]
        if time_shift_random:
          if ticks[0]<desired_samples+time_shift_samples or \
             ticks[1]>(nsamples-desired_samples-time_shift_samples):
            continue
        else:
          if ticks[0]<desired_samples+time_shift_samples or \
             ticks[1]>(nsamples-desired_samples+time_shift_samples):
            continue
        all_words[word] = True
        if wavfile in validation_files:
          set_index = 'validation'
        elif wavfile in testing_files:
          set_index = 'testing'
        elif word in partition_words:
          if wavfile in partition_validation_files:
            set_index = 'validation'
          elif wavfile in partition_training_files:
            set_index = 'training'
          else:
            continue
        else:
          set_index = which_set(annotation[0]+annotation[1]+annotation[2],
                                validation_percentage, validation_offset_percentage, \
                                testing_percentage)
        # If it's a known class, store its detail
        if word in wanted_words_index:
          self.data_index[set_index].append({'label': word, 'file': wav_path, \
                                             'ticks': ticks, 'kind': kind})
    if not all_words:
      print('WARNING: No wanted words found in labels')
    if validation_percentage+testing_percentage<100:
      for index, wanted_word in enumerate(wanted_words):
        if wanted_word not in all_words:
          print('WARNING: '+wanted_word+' not in labels')
    # equalize
    for set_index in ['validation', 'testing', 'training']:
      print('num %s labels' % set_index)
      words = [sample['label'] for sample in self.data_index[set_index]]
      if set_index != 'testing':
        for uniqword in sorted(set(words)):
          print('%8d %s' % (sum([word==uniqword for word in words]), uniqword))
      if set_index == 'validation' or len(self.data_index[set_index])==0:
        continue
      word_indices = {}
      for isample in range(len(self.data_index[set_index])):
        sample = self.data_index[set_index][isample]
        if sample['label'] in word_indices:
          word_indices[sample['label']].append(isample)
        else:
          word_indices[sample['label']]=[isample]
      if set_index == 'training':
        samples_largest = max([len(word_indices[x]) for x in word_indices.keys()])
        for word in sorted(list(word_indices.keys())):
          samples_have = len(word_indices[word])
          samples_needed = samples_largest - samples_have
          for _ in range(samples_needed):
            add_this = word_indices[word][random.randrange(samples_have)]
            self.data_index[set_index].append(self.data_index[set_index][add_this])
      elif set_index == 'testing':
        if testing_equalize_ratio>0:
          samples_smallest = min([len(word_indices[x]) for x in word_indices.keys()])
          del_these = []
          for word in sorted(list(word_indices.keys())):
            samples_have = len(word_indices[word])
            samples_needed = min(samples_have, testing_equalize_ratio * samples_smallest)
            if samples_needed<samples_have:
              del_these.extend(random.sample(word_indices[word], \
                               samples_have-samples_needed))
          for i in sorted(del_these, reverse=True):
            del self.data_index[set_index][i]
        if testing_max_samples>0 and testing_max_samples<len(self.data_index[set_index]):
          self.data_index[set_index] = random.sample(self.data_index[set_index], \
                                                     testing_max_samples)
      if set_index == 'testing':
        words = [sample['label'] for sample in self.data_index[set_index]]
        for uniqword in sorted(set(words)):
          print('%7d %s' % (sum([word==uniqword for word in words]), uniqword))
    # Make sure the ordering is random.
    for set_index in ['validation', 'testing', 'training']:
      random.shuffle(self.data_index[set_index])
    # Prepare the rest of the result data structure.
    self.words_list = wanted_words
    self.word_to_index = {}
    for word in all_words:
      if word in wanted_words_index:
        self.word_to_index[word] = wanted_words_index[word]

  def prepare_processing_graph(self, model_settings):
    """Builds a TensorFlow graph to apply the input distortions.

    Creates a graph that loads a WAVE file, decodes it, scales the volume,
    shifts it in time, calculates a spectrogram, and
    then builds an MFCC fingerprint from that.

    This must be called with an active TensorFlow session running, and it
    creates multiple placeholder inputs, and one output:

      - wav_filename_placeholder_: Filename of the WAV to load.
      - foreground_volume_placeholder_: How loud the main clip should be.
      - time_shift_offset_placeholder_: How much to move the clip in time.
      - mfcc_: Output 2D fingerprint of processed audio.

    Args:
      model_settings: Information about the current model being trained.
    """
    self.waveform_ = scale_foreground
    self.spectrogram_ = compute_spectrograms
    self.mfcc_ = compute_mfccs

  def set_size(self, mode):
    """Calculates the number of samples in the dataset partition.

    Args:
      mode: Which partition, must be 'training', 'validation', or 'testing'.

    Returns:
      Number of samples in the partition.
    """
    return len(self.data_index[mode])

  def get_data(self, how_many, offset, model_settings, 
               time_shift_ms, time_shift_random, mode):
    """Gather samples from the data set, applying transformations as needed.

    When the mode is 'training', a random selection of samples will be returned,
    otherwise the first N clips in the partition will be used. This ensures that
    validation always uses the same samples, reducing noise in the metrics.

    Args:
      how_many: Desired number of samples to return. -1 means the entire
        contents of this partition.
      offset: Where to start when fetching deterministically.
      model_settings: Information about the current model being trained.
      time_shift: How much to randomly shift the clips by in time.
      time_shift_random:  True means to pick a random shift; False means shift by exactly this value
      mode: Which partition to use, must be 'training', 'validation', or
        'testing'.
      sess: TensorFlow session that was active when processor was created.

    Returns:
      List of sample data for the transformed samples, and list of label indexes
    """
    time_shift_samples = int((time_shift_ms * model_settings["sample_rate"]) / 1000)
    # Pick one of the partitions to choose samples from.
    candidates = self.data_index[mode]
    ncandidates = len(self.data_index[mode])
    if how_many == -1:
      sample_count = ncandidates
    else:
      sample_count = max(0, min(how_many, ncandidates - offset))
    samples = []
    desired_samples = model_settings['desired_samples']
    channel_count = model_settings['channel_count']
    pick_deterministically = (mode != 'training')
    if model_settings['representation']=='waveform':
      input_to_use = self.waveform_
    elif model_settings['representation']=='spectrogram':
      input_to_use = self.spectrogram_
    elif model_settings['representation']=='mel-cepstrum':
      input_to_use = self.mfcc_
    foreground_indexed = np.zeros((sample_count, channel_count, desired_samples),
                                  dtype=np.float32)
    labels = np.zeros(sample_count, dtype=np.int)
    # repeatedly to generate the final output sample data we'll use in training.
    for i in range(offset, offset + sample_count):
      # Pick which audio sample to use.
      if how_many == -1 or pick_deterministically:
        sample_index = i
        sample = candidates[sample_index]
      else:
        sample_index = np.random.randint(len(candidates))
        sample = candidates[sample_index]

      foreground_offset = (np.random.randint(sample['ticks'][0], 1+sample['ticks'][1]) if
            sample['ticks'][0] < sample['ticks'][1] else sample['ticks'][0])
      sample_rate, song = spiowav.read(sample['file'], mmap=True)
      if np.ndim(song)==1:
        song = np.expand_dims(song, axis=1)
      nchannels = np.shape(song)[1]
      assert sample_rate == model_settings['sample_rate']
      assert nchannels == channel_count
      if time_shift_samples > 0:
        if time_shift_random:
          time_shift_amount = np.random.randint(-time_shift_samples, time_shift_samples)
        else:
          time_shift_amount = time_shift_samples
      else:
        time_shift_amount = 0
      foreground_clipped = song[foreground_offset-desired_samples//2 - time_shift_amount :
                                foreground_offset+desired_samples//2 - time_shift_amount,
                                :]
      foreground_float32 = foreground_clipped.astype(np.float32)
      foreground_scaled = foreground_float32 / abs(np.iinfo(np.int16).min) #extreme
      foreground_indexed[i - offset,:,:] = foreground_scaled.transpose()
      label_index = self.word_to_index[sample['label']]
      labels[i - offset] = label_index
      samples.append(sample)
    # Run the graph to produce the output audio.
    data = tf.reshape(input_to_use(foreground_indexed, 1.0, model_settings),
                      [sample_count, -1])
    return data, labels, samples
