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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from tensorflow.python.ops import gen_audio_ops as audio_ops
from tensorflow.python.ops import io_ops
from tensorflow.python.platform import gfile
from tensorflow.python.util import compat

MAX_NUM_WAVS_PER_CLASS = 2**27 - 1  # ~134M
SILENCE_LABEL = '_silence_'
SILENCE_INDEX = 1
UNKNOWN_WORD_LABEL = 'other'
UNKNOWN_WORD_INDEX = 0
BACKGROUND_NOISE_DIR_NAME = '_background_noise_'


def prepare_words_list(wanted_words, silence_percentage, unknown_percentage):
  """Prepends common tokens to the custom word list.

  Args:
    wanted_words: List of strings containing the custom words.

  Returns:
    List with the standard silence and unknown tokens added.
  """
  words_list=[]
  if silence_percentage>0.0:
    return words_list.append(SILENCE_LABEL)
  if unknown_percentage>0.0:
    return words_list.append(UNKNOWN_WORD_LABEL)
  return words_list + wanted_words


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
  hash_name_hashed = hashlib.sha1(compat.as_bytes(hash_name)).hexdigest()
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

  def __init__(self, data_url, data_dir, silence_percentage, unknown_percentage,
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
    self.prepare_data_index(silence_percentage, unknown_percentage,
                            time_shift_ms, time_shift_random,
                            wanted_words, labels_touse,
                            validation_percentage, validation_offset_percentage, validation_files,
                            testing_percentage, testing_files, subsample_skip, subsample_word,
                            partition_word, partition_n, partition_training_files, partition_validation_files,
                            testing_equalize_ratio, testing_max_samples,
                            model_settings)
    self.prepare_background_data()
    self.prepare_processing_graph(model_settings)

  def prepare_data_index(self, silence_percentage, unknown_percentage,
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
      silence_percentage: How much of the resulting data should be background.
      unknown_percentage: How much should be audio outside the wanted classes.
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
    # Make sure the shuffling and picking of unknowns is deterministic.
    wanted_words_index = {}
    for index, wanted_word in enumerate(wanted_words):
      wanted_words_index[wanted_word] = index
      if silence_percentage>0.0:
        wanted_words_index[wanted_word] += 1
      if unknown_percentage>0.0:
        wanted_words_index[wanted_word] += 1
    self.data_index = {'validation': [], 'testing': [], 'training': []}
    unknown_index = {'validation': [], 'testing': [], 'training': []}
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
    for csv_path in gfile.Glob(search_path):
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
        # Treat the '_background_noise_' folder as a special case, since we expect
        # it to contain long audio samples we mix in to improve training.
        if word == BACKGROUND_NOISE_DIR_NAME:
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
        # If it's a known class, store its detail, otherwise add it to the list
        # we'll use to train the unknown label.
        if word in wanted_words_index:
          self.data_index[set_index].append({'label': word, 'file': wav_path, \
                                             'ticks': ticks, 'kind': kind})
        else:
          unknown_index[set_index].append({'label': word, 'file': wav_path, \
                                           'ticks': ticks, 'kind': kind})
    if not all_words:
      print('WARNING: No wanted words found in labels')
    if validation_percentage+testing_percentage<100:
      for index, wanted_word in enumerate(wanted_words):
        if wanted_word not in all_words:
          print('WARNING: '+wanted_word+' not in labels')
    # equalize
    for set_index in ['validation', 'testing', 'training']:
      tf.logging.info('num %s labels', set_index)
      words = [sample['label'] for sample in self.data_index[set_index]]
      if set_index != 'testing':
        for uniqword in sorted(set(words)):
          tf.logging.info('%8d %s', sum([word==uniqword for word in words]), uniqword)
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
          tf.logging.info('%7d %s', sum([word==uniqword for word in words]), uniqword)
    # We need an arbitrary file to load as the input for the silence samples.
    # It's multiplied by zero later, so the content doesn't matter.
    if len(self.data_index['training'])>0:
      silence_wav_path = self.data_index['training'][0]['file']
    elif len(self.data_index['testing'])>0:
      silence_wav_path = self.data_index['testing'][0]['file']
    elif len(self.data_index['validation'])>0:
      silence_wav_path = self.data_index['validation'][0]['file']
    for set_index in ['validation', 'testing', 'training']:
      set_size = len(self.data_index[set_index])
      silence_size = int(math.ceil(set_size * silence_percentage / 100))
      for _ in range(silence_size):
        self.data_index[set_index].append({
            'label': SILENCE_LABEL,
            'file': silence_wav_path
        })
      # Pick some unknowns to add to each partition of the data set.
      unknown_needed = int(math.ceil(set_size * unknown_percentage / 100))
      unknown_have = range(len(unknown_index[set_index]))
      for _ in range(unknown_needed):
        add_this = random.choice(unknown_have)
        self.data_index[set_index].append(unknown_index[set_index][add_this])
    # Make sure the ordering is random.
    for set_index in ['validation', 'testing', 'training']:
      random.shuffle(self.data_index[set_index])
    # Prepare the rest of the result data structure.
    self.words_list = prepare_words_list(wanted_words, silence_percentage, \
                                         unknown_percentage)
    self.word_to_index = {}
    for word in all_words:
      if word in wanted_words_index:
        self.word_to_index[word] = wanted_words_index[word]
      else:
        self.word_to_index[word] = UNKNOWN_WORD_INDEX
    if silence_percentage>0.0:
      self.word_to_index[SILENCE_LABEL] = SILENCE_INDEX
    if unknown_percentage>0.0:
      self.word_to_index[UNKNOWN_WORD_LABEL] = UNKNOWN_WORD_INDEX

  def prepare_background_data(self):
    """Searches a folder for background noise audio, and loads it into memory.

    It's expected that the background audio samples will be in a subdirectory
    named '_background_noise_' inside the 'data_dir' folder, as .wavs that match
    the sample rate of the training data, but can be much longer in duration.

    If the '_background_noise_' folder doesn't exist at all, this isn't an
    error, it's just taken to mean that no background noise augmentation should
    be used. If the folder does exist, but it's empty, that's treated as an
    error.

    Returns:
      List of raw PCM-encoded audio samples of background noise.

    Raises:
      Exception: If files aren't found in the folder.
    """

    ### need to refactor this to use csv files if background noise is used in future

    self.background_data = []
    background_dir = os.path.join(self.data_dir, BACKGROUND_NOISE_DIR_NAME)
    if not os.path.exists(background_dir):
      return self.background_data
    with tf.Session(graph=tf.Graph()) as sess:
      wav_filename_placeholder = tf.placeholder(tf.string, [])
      wav_loader = io_ops.read_file(wav_filename_placeholder)
      wav_decoder = audio_ops.decode_wav(wav_loader, desired_channels=1)
      search_path = os.path.join(self.data_dir, BACKGROUND_NOISE_DIR_NAME,
                                 '*.wav')
      for wav_path in gfile.Glob(search_path):
        wav_data = sess.run(
            wav_decoder,
            feed_dict={wav_filename_placeholder: wav_path}).audio.flatten()
        self.background_data.append(wav_data)
      if not self.background_data:
        raise Exception('No background wav files were found in ' + search_path)

  def prepare_processing_graph(self, model_settings):
    """Builds a TensorFlow graph to apply the input distortions.

    Creates a graph that loads a WAVE file, decodes it, scales the volume,
    shifts it in time, adds in background noise, calculates a spectrogram, and
    then builds an MFCC fingerprint from that.

    This must be called with an active TensorFlow session running, and it
    creates multiple placeholder inputs, and one output:

      - wav_filename_placeholder_: Filename of the WAV to load.
      - foreground_volume_placeholder_: How loud the main clip should be.
      - time_shift_offset_placeholder_: How much to move the clip in time.
      - background_data_placeholder_: PCM sample data for background noise.
      - background_volume_placeholder_: Loudness of mixed-in background.
      - mfcc_: Output 2D fingerprint of processed audio.

    Args:
      model_settings: Information about the current model being trained.
    """
    desired_samples = model_settings['desired_samples']
    channel_count = model_settings['channel_count']
    sample_rate = model_settings['sample_rate']
    self.foreground_data_placeholder_ = tf.placeholder(tf.float32,
                                                       [desired_samples, channel_count])
    # Allow the audio sample's volume to be adjusted.
    self.foreground_volume_placeholder_ = tf.placeholder(tf.float32, [])
    scaled_foreground = tf.multiply(self.foreground_data_placeholder_,
                                    self.foreground_volume_placeholder_)
    # Mix in background noise.
    self.background_data_placeholder_ = tf.placeholder(tf.float32,
                                                       [desired_samples, channel_count])
    self.background_volume_placeholder_ = tf.placeholder(tf.float32, [])
    background_mul = tf.multiply(self.background_data_placeholder_,
                                 self.background_volume_placeholder_)
    background_add = tf.add(background_mul, scaled_foreground)
    background_clamp = tf.clip_by_value(background_add, -1.0, 1.0)
    # Run the spectrogram and MFCC ops to get a 2D 'fingerprint' of the audio.
    self.waveform_ = background_clamp
    spectrograms = []
    for ichannel in range(channel_count):
      spectrograms.append(audio_ops.audio_spectrogram(
          tf.slice(background_clamp, [0, ichannel], [-1, 1]),
          window_size=model_settings['window_size_samples'],
          stride=model_settings['window_stride_samples'],
          magnitude_squared=True))
    self.spectrogram_ = tf.stack(spectrograms, -1)
    mfccs = []
    for ichannel in range(channel_count):
      mfccs.append(audio_ops.mfcc(
          spectrograms[ichannel],
          sample_rate,
          upper_frequency_limit=model_settings['sample_rate']//2,
          filterbank_channel_count=model_settings['filterbank_channel_count'],
          dct_coefficient_count=model_settings['dct_coefficient_count']))
    self.mfcc_ = tf.stack(mfccs, -1)


  def set_size(self, mode):
    """Calculates the number of samples in the dataset partition.

    Args:
      mode: Which partition, must be 'training', 'validation', or 'testing'.

    Returns:
      Number of samples in the partition.
    """
    return len(self.data_index[mode])

  def get_data(self, how_many, offset, model_settings, background_frequency,
               background_volume_range, time_shift_ms, time_shift_random, mode, sess):
    """Gather samples from the data set, applying transformations as needed.

    When the mode is 'training', a random selection of samples will be returned,
    otherwise the first N clips in the partition will be used. This ensures that
    validation always uses the same samples, reducing noise in the metrics.

    Args:
      how_many: Desired number of samples to return. -1 means the entire
        contents of this partition.
      offset: Where to start when fetching deterministically.
      model_settings: Information about the current model being trained.
      background_frequency: How many clips will have background noise, 0.0 to
        1.0.
      background_volume_range: How loud the background noise will be.
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
    # Data and labels will be populated and returned.
    data = np.zeros((sample_count, model_settings['fingerprint_size']))
    labels = np.zeros(sample_count)
    samples = []
    desired_samples = model_settings['desired_samples']
    use_background = self.background_data and (mode == 'training')
    pick_deterministically = (mode != 'training')
    if model_settings['representation']=='waveform':
      input_to_use = self.waveform_
    elif model_settings['representation']=='spectrogram':
      input_to_use = self.spectrogram_
    elif model_settings['representation']=='mel-cepstrum':
      input_to_use = self.mfcc_
    # Use the processing graph we created earlier to repeatedly to generate the
    # final output sample data we'll use in training.
    for i in xrange(offset, offset + sample_count):
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
      assert nchannels == model_settings['channel_count']
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
      foreground_reshaped = foreground_scaled.reshape([nchannels, desired_samples], order='F')
      foreground_indexed = foreground_reshaped.reshape([desired_samples,nchannels])
      input_dict = { self.foreground_data_placeholder_: foreground_indexed }
      # Choose a section of background noise to mix in.
      if use_background or sample['label'] == SILENCE_LABEL:
        background_index = np.random.randint(len(self.background_data))
        background_samples = self.background_data[background_index]
        background_offset = np.random.randint(
            0, len(background_samples) - desired_samples*nchannels)
        background_clipped = background_samples[background_offset:(
            background_offset + desired_samples*nchannels)]
        background_reshaped = background_clipped.reshape([desired_samples, nchannels])
        if sample['label'] == SILENCE_LABEL:
          background_volume = np.random.uniform(0, 1)
        elif np.random.uniform(0, 1) < background_frequency:
          background_volume = np.random.uniform(0, background_volume_range)
        else:
          background_volume = 0
      else:
        background_reshaped = np.zeros([desired_samples, nchannels])
        background_volume = 0
      input_dict[self.background_data_placeholder_] = background_reshaped
      input_dict[self.background_volume_placeholder_] = background_volume
      # If we want silence, mute out the main sample but leave the background.
      if sample['label'] == SILENCE_LABEL:
        input_dict[self.foreground_volume_placeholder_] = 0
      else:
        input_dict[self.foreground_volume_placeholder_] = 1
      # Run the graph to produce the output audio.
      data[i - offset, :] = sess.run(input_to_use, feed_dict=input_dict).flatten()
      label_index = self.word_to_index[sample['label']]
      labels[i - offset] = label_index
      samples.append(sample)
    return data, labels, samples
