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
import re
import sys
import csv
from glob import glob

import numpy as np

import tifffile

import signal
from multiprocessing import Process, Queue, cpu_count

import importlib

MAX_NUM_WAVS_PER_CLASS = 2**27 - 1  # ~134M

queues = {}
processes = {}
offsets = {}

def term(signum, frame):
    for ps in processes.values():
        for p in ps:
            p.kill()
    sys.exit()

def which_set(filename, validation_percentage, validation_offset_percentage, testing_percentage):
    """Determines which data partition the file should belong to.

    We want to keep files in the same training, validation, or testing sets even
    if new ones are added over time. This makes it less likely that testing
    sounds will accidentally be reused in training when long runs are restarted
    for example. To keep this stability, a hash of the filename is taken and used
    to determine which set it should belong to. This determination only depends on
    the name and the set proportions, so it won't change as other files are added.

    It's also useful to associate particular files as related (for example words
    spoken by the same person), so anything after '_nohash_' in a filename is
    ignored for set determination. This ensures that 'bobby_nohash_0.wav' and
    'bobby_nohash_1.wav' are always in the same set, for example.

    Args:
      filename: File path of the sound.
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
    hash_name_hashed = hashlib.sha1(bytes(hash_name,"utf-8")).hexdigest()
    percentage_hash = ((int(hash_name_hashed, 16) %
                        (MAX_NUM_WAVS_PER_CLASS + 1)) *
                       (100.0 / MAX_NUM_WAVS_PER_CLASS))
    if percentage_hash < testing_percentage:
        result = 'testing'
    elif (testing_percentage + validation_offset_percentage) < \
         percentage_hash < \
         (testing_percentage + validation_offset_percentage + validation_percentage):
        result = 'validation'
    else:
        result = 'training'
    return result

class AudioProcessor(object):
    """Handles loading, partitioning, and preparing audio training data."""

    def __init__(self, data_dir,
                 shiftby_ms,
                 labels_touse, kinds_touse,
                 validation_percentage, validation_offset_percentage, validation_files,
                 testing_percentage, testing_files, subsample_skip, subsample_label,
                 partition_label, partition_n, partition_training_files, partition_validation_files,
                 random_seed_batch,
                 testing_equalize_ratio, testing_max_sounds,
                 model_settings, loss, overlapped_prefix,
                 queue_size, max_procs,
                 use_audio, use_video, video_findfile, video_bkg_frames,
                 audio_read_plugin, video_read_plugin,
                 audio_read_plugin_kwargs, video_read_plugin_kwargs):
        self.data_dir = data_dir
        self.np_rng = np.random.default_rng(None if random_seed_batch==-1 else random_seed_batch)

        sys.path.append(os.path.dirname(audio_read_plugin))
        self.audio_read_plugin = os.path.basename(audio_read_plugin)
        self.audio_read_plugin_kwargs = audio_read_plugin_kwargs

        sys.path.append(os.path.dirname(video_read_plugin))
        self.video_read_plugin = os.path.basename(video_read_plugin)
        self.video_read_plugin_kwargs = video_read_plugin_kwargs

        self.prepare_data_index(shiftby_ms,
                                labels_touse, kinds_touse,
                                validation_percentage, validation_offset_percentage, validation_files,
                                testing_percentage, testing_files, subsample_skip, subsample_label,
                                partition_label, partition_n, partition_training_files, partition_validation_files,
                                testing_equalize_ratio, testing_max_sounds,
                                model_settings, loss, overlapped_prefix,
                                use_audio, use_video, video_findfile, video_bkg_frames)
        self.queue_size = queue_size
        self.max_procs = max_procs

        signal.signal(signal.SIGTERM, term)

    def audio_read(self, fullpath, start_tic=None, stop_tic=None):
        audio_read_module = importlib.import_module(self.audio_read_plugin)
        return audio_read_module.audio_read(fullpath, start_tic, stop_tic,
                                            **self.audio_read_plugin_kwargs)

    def video_read(self, fullpath, start_frame=None, stop_frame=None):
        video_read_module = importlib.import_module(self.video_read_plugin)
        return video_read_module.video_read(fullpath, start_frame, stop_frame,
                                            **self.video_read_plugin_kwargs)

    def catalog_overlaps(self, data):
        data.sort(key=lambda x: x['ticks'][0])
        k = [True]*len(data)
        for i in range(len(data)):
            for j in range(i-1):
                if not k[j]:  continue
                if data[j]['ticks'][1] < data[i]['ticks'][0]:
                    k[j] = False
                    continue
                data[i]['overlaps'].append(j)

    def prepare_data_index(self,
                           shiftby_ms,
                           labels_touse, kinds_touse,
                           validation_percentage, validation_offset_percentage, validation_files,
                           testing_percentage, testing_files, subsample_skip, subsample_label,
                           partition_label, partition_n, partition_training_files, partition_validation_files,
                           testing_equalize_ratio, testing_max_sounds,
                           model_settings, loss, overlapped_prefix, use_audio, use_video,
                           video_findfile, video_bkg_frames):
        """Prepares a list of the sounds organized by set and label.

        The training loop needs a list of all the available data, organized by
        which partition it should belong to, and with ground truth labels attached.
        This function analyzes the folders below the `data_dir`, figures out the
        right
        labels for each file based on the name of the subdirectory it belongs to,
        and uses a stable hash to assign it to a data set partition.

        Args:
          labels_touse: Labels of the classes we want to be able to recognize.
          validation_percentage: How much of the data set to use for validation.
          validation_offset_percentage: Which part of the data set to use for validation.
          testing_percentage: How much of the data set to use for testing.

        Returns:
          Dictionary containing a list of file information for each set partition,
          and a lookup map for each class to determine its numeric index.

        Raises:
          Exception: If expected files are not found.
        """
        self.data_index = {'validation': [], 'testing': [], 'training': []}
        all_labels = {}
        # Look through all the subfolders to find sounds
        context_tics = int(model_settings['audio_tic_rate'] * model_settings['context_ms'] / 1000)
        video_frame_rate = model_settings['video_frame_rate']
        video_frame_width = model_settings['video_frame_width']
        video_frame_height = model_settings['video_frame_height']
        video_channels = model_settings['video_channels']
        shiftby_tics = int(shiftby_ms * model_settings["audio_tic_rate"] / 1000)
        search_path = os.path.join(self.data_dir, '*', '*.csv')
        audio_ntics = {}
        video_nframes = {}
        subsample = {x:int(y) for x,y in zip(subsample_label.split(','),subsample_skip.split(','))
                              if x != ''}
        partition_labels = partition_label.split(',')
        if '' in partition_labels:
            partition_labels.remove('')
        for csv_path in glob(search_path):
            with (open(csv_path, 'r')) as csv_file:
                annotation_reader = csv.reader(csv_file)
                annotation_list = list(annotation_reader)
            if len(partition_labels)>0:
                self.np_rng.shuffle(annotation_list)
            for (iannotation, annotation) in enumerate(annotation_list):
                wavfile=annotation[0]
                ticks=[int(annotation[1]),int(annotation[2])]
                if ticks[0]>ticks[1]:
                    print("ERROR: "+str(annotation)+" has start tic after stop tic")
                    continue
                kind=annotation[3]
                label=annotation[4]
                if kind not in kinds_touse:
                    continue
                if (label if loss=='exclusive' else
                    label.removeprefix(overlapped_prefix)) not in labels_touse:
                    continue
                wav_path=os.path.join(os.path.dirname(csv_path),wavfile)
                wav_base2=[os.path.basename(os.path.dirname(csv_path)), wavfile]
                if wavfile in validation_files:
                    set_index = 'validation'
                elif wavfile in testing_files:
                    set_index = 'testing'
                elif label in partition_labels:
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
                if label in subsample and iannotation % subsample[label] != 0 and set_index=='training':
                    continue
                if label in partition_labels:
                    if wavfile not in partition_training_files and \
                       wavfile not in partition_validation_files:
                        continue
                    if wavfile in partition_training_files and \
                       sum(x['label']==label and x['file']==wav_base2 \
                           for x in self.data_index['training']) >= partition_n:
                        continue
                if use_audio and wav_path not in audio_ntics:
                    audio_tic_rate, audio_data_shape, _ = self.audio_read(wav_path, 0, 1)
                    if audio_tic_rate != model_settings['audio_tic_rate']:
                        print(f'ERROR: audio_tic_rate is set to {model_settings["audio_tic_rate"]} '
                              f'in configuration.py but is actually {audio_tic_rate} in {wav_path}')
                    if audio_data_shape[1] != model_settings['audio_nchannels']:
                        print(f'ERROR: audio_nchannels is set to {model_settings["audio_nchannels"]} '
                              f'in configuration.py but is actually {audio_data_shape[1]} in {wav_path}')
                    audio_ntics[wav_path] = audio_data_shape[0]
                if use_audio:
                    if ticks[1] < context_tics//2 + shiftby_tics or \
                       ticks[0] > (audio_ntics[wav_path] - context_tics//2 + shiftby_tics):
                        print(f"WARNING: {str(annotation)} is too close to both edges of recording.  "
                              f"not using at all")
                        continue
                    if ticks[0] < context_tics//2 + shiftby_tics:
                        print(f"WARNING: {str(annotation)} is close to beginning of recording.  "
                              f"shortening interval to usable range")
                        ticks[0] = context_tics//2 + shiftby_tics
                    if ticks[1] > audio_ntics[wav_path] - context_tics//2 + shiftby_tics:
                        print(f"WARNING: {str(annotation)} is close to end of recording.  "
                              f"shortening interval to usable range")
                        ticks[1] = audio_ntics[wav_path] - context_tics//2 + shiftby_tics
                if use_video and wav_path not in video_nframes:
                    sound_dirname = os.path.join(self.data_dir, wav_base2[0])
                    vidfile = video_findfile(sound_dirname, wavfile)
                    if not vidfile:
                        print(f"ERROR: video file corresponding to {wavfile} not found")
                        continue
                    frame_rate, video_shape, _, video_data = self.video_read(os.path.join(sound_dirname,vidfile))
                    if video_frame_rate != frame_rate:
                        print(f'ERROR: video_frame_rate is set to {video_frame_rate} '
                              f'in configuration.py but is actually {frame_rate} in {vidfile}')
                        continue
                    if video_frame_width != video_shape[1]:
                        print(f'ERROR: video_frame_width is set to {video_frame_width} '
                              f'in configuration.py but is actually {video_shape[1]} in {vidfile}')
                        continue
                    if video_frame_height != video_shape[2]:
                        print(f'ERROR: video_frame_height is set to {video_frame_height} '
                              f'in configuration.py but is actually {video_shape[2]} in {vidfile}')
                        continue
                    if max(video_channels) > video_shape[3]:
                        print(f'ERROR: video_channels is set to {video_channels} '
                              f'in configuration.py but {vidfile} has only {video_shape[3]} channels')
                        continue
                    video_nframes[wav_path] = video_shape[0]

                    tiffile = os.path.join(sound_dirname, os.path.splitext(vidfile)[0]+".tif")
                    if not os.path.exists(tiffile):
                        compute_background(vidfile, video_bkg_frames, video_data, tiffile)

                if use_video:
                    if ticks[1] < context_tics//2 + shiftby_tics or \
                       ticks[0] > video_nframes[wav_path] / video_frame_rate * model_settings['audio_tic_rate'] - context_tics//2 + shiftby_tics:
                        print(f"WARNING: {str(annotation)} is too close to both edges of recording.  "
                              f"not using at all")
                        continue
                    if ticks[0] < context_tics//2 + shiftby_tics:
                        print(f"WARNING: {str(annotation)} is close to beginning of recording.  "
                              f"shortening interval to usable range")
                        ticks[0] = context_tics//2 + shiftby_tics
                    if ticks[1] > video_nframes[wav_path] / video_frame_rate * model_settings['audio_tic_rate'] - context_tics//2 + shiftby_tics:
                        print(f"WARNING: {str(annotation)} is close to end of recording.  "
                              f"shortening interval to usable range")
                        ticks[1] = video_nframes[wav_path] / video_frame_rate * model_settings['audio_tic_rate'] - context_tics//2 + shiftby_tics
                all_labels[label] = True
                self.data_index[set_index].append({'label': label,
                                                   'file': wav_base2, \
                                                   'ticks': ticks,
                                                   'kind': kind,
                                                   'overlaps': []})
        if not all_labels:
            print('WARNING: None of the labels to use were found in the ground truth')
        if validation_percentage+testing_percentage<100:
            for label_touse in labels_touse:
                if label_touse not in all_labels:
                    print('WARNING: '+label_touse+' not in ground truth')
                if loss=='overlapped':
                    if overlapped_prefix+label_touse not in all_labels:
                        print('WARNING: '+overlapped_prefix+label_touse+' not in ground truth')
        # equalize
        for set_index in ['validation', 'testing', 'training']:
            print("num "+set_index+" labels")
            if set_index != 'testing':
                self.catalog_overlaps(self.data_index[set_index])
                labels = [sound['label'] for sound in self.data_index[set_index]]
                for uniqlabel in sorted(set(labels)):
                    print('%8d %s' % (sum(label==uniqlabel for label in labels), uniqlabel))
            if set_index == 'validation' or len(self.data_index[set_index])==0: continue
            label_kind_indices = {}
            for isound in range(len(self.data_index[set_index])):
                sound = self.data_index[set_index][isound]
                if (sound['label'], sound['kind']) in label_kind_indices:
                    label_kind_indices[(sound['label'], sound['kind'])].append(isound)
                else:
                    label_kind_indices[(sound['label'], sound['kind'])]=[isound]
            if set_index == 'training':
                sounds_largest = max(len(x) for x in label_kind_indices.values())
                for label in sorted(list(label_kind_indices.keys())):
                    sounds_have = len(label_kind_indices[label])
                    sounds_needed = sounds_largest - sounds_have
                    for _ in range(sounds_needed):
                        add_this = label_kind_indices[label][self.np_rng.integers(sounds_have)]
                        self.data_index[set_index].append(self.data_index[set_index][add_this])
            elif set_index == 'testing':
                if testing_equalize_ratio>0:
                    sounds_smallest = min(filter(lambda x: x!=0,
                                                 [len(x) for x in label_kind_indices.values()]))

                    del_these = []
                    for label in sorted(list(label_kind_indices.keys())):
                        sounds_have = len(label_kind_indices[label])
                        sounds_needed = min(sounds_have, testing_equalize_ratio * sounds_smallest)
                        if sounds_needed<sounds_have:
                            del_these.extend(self.np_rng.choice(label_kind_indices[label], \
                                                                sounds_have-sounds_needed,
                                                                replace=False))
                    for i in sorted(del_these, reverse=True):
                        del self.data_index[set_index][i]
                if 0 < testing_max_sounds < len(self.data_index[set_index]):
                    self.data_index[set_index] = self.np_rng.choice(self.data_index[set_index], \
                                                                    testing_max_sounds,
                                                                    replace=False).tolist()
            if set_index == 'testing':
                self.catalog_overlaps(self.data_index['testing'])
                labels = [sound['label'] for sound in self.data_index['testing']]
                for uniqlabel in sorted(set(labels)):
                    print('%7d %s' % (sum(label==uniqlabel for label in labels), uniqlabel))
        # Make sure the ordering is random.
        for set_index in ['validation', 'testing', 'training']:
            self.np_rng.shuffle(self.data_index[set_index])
        # Prepare the rest of the result data structure.
        self.labels_list = labels_touse

    def set_size(self, mode):
        """Calculates the number of sounds in the dataset partition.

        Args:
          mode: Which partition, must be 'training', 'validation', or 'testing'.

        Returns:
          Number of sounds in the partition.
        """
        return len(self.data_index[mode])

    def _get_data(self, q, o, how_many, offset, model_settings, loss, overlapped_prefix,
                  shiftby_ms, mode, use_audio, use_video, video_findfile):
        q.cancel_join_thread()
        while True:
            # Pick one of the partitions to choose sounds from.
            pick_deterministically = mode != 'training'
            if pick_deterministically:
                offset = o.get()
            candidates = self.data_index[mode]
            ncandidates = len(candidates)
            nsounds = min(how_many, ncandidates - offset)
            sounds = []
            context_tics = int(model_settings['audio_tic_rate'] * model_settings['context_ms'] / 1000)
            audio_tic_rate  = model_settings['audio_tic_rate']
            audio_nchannels = model_settings['audio_nchannels']
            video_frame_rate = model_settings['video_frame_rate']
            video_channels = model_settings['video_channels']
            shiftby_tics = int(shiftby_ms * audio_tic_rate / 1000)
            if use_audio:
                audio_slice = np.zeros((nsounds, context_tics, audio_nchannels), dtype=np.float32)
            if use_video:
                nframes = round(model_settings['context_ms'] / 1000 * video_frame_rate)
                video_slice = np.zeros((nsounds,
                                        nframes,
                                        model_settings['video_frame_height'],
                                        model_settings['video_frame_width'],
                                        len(video_channels)),
                                       dtype=np.float32)
                bkg = {}
            if loss=='exclusive':
                labels = np.zeros(nsounds, dtype=np.int32)
            else:
                labels = 2*np.ones((nsounds, len(self.labels_list)), dtype=np.float32)
            # repeatedly to generate the final output sound data we'll use in training.
            for i in range(offset, offset + nsounds):
                # Pick which sound to use.
                if pick_deterministically:
                    isound = i
                else:
                    isound = self.np_rng.integers(ncandidates)
                sound = candidates[isound]

                offset_tic = (self.np_rng.integers(sound['ticks'][0], high=1+sound['ticks'][1]) \
                              if sound['ticks'][0] < sound['ticks'][1] \
                              else sound['ticks'][0])
                start_tic = offset_tic - math.floor(context_tics/2) - shiftby_tics
                stop_tic  = offset_tic + math.ceil(context_tics/2) - shiftby_tics
                if use_audio:
                    wavpath = os.path.join(self.data_dir, *sound['file'])
                    _, _, audio_data = self.audio_read(wavpath, start_tic, stop_tic)
                    audio_slice[i-offset,:,:] = audio_data.astype(np.float32) / abs(np.iinfo(np.int16).min)
                if use_video:
                    sound_basename = sound['file'][1]
                    sound_dirname = os.path.join(self.data_dir, sound['file'][0])
                    vidfile = video_findfile(sound_dirname, sound_basename)
                    tiffile = os.path.join(sound_dirname, os.path.splitext(vidfile)[0]+".tif")
                    if vidfile not in bkg:
                        bkg[vidfile] = tifffile.imread(tiffile)
                    start_frame = round(start_tic / audio_tic_rate * video_frame_rate)
                    _, video_data = self.video_read(os.path.join(sound_dirname, vidfile),
                                                    start_frame, start_frame+nframes)
                    for iframe, frame in enumerate(video_data):
                        video_slice[i-offset,iframe,:,:,:] = \
                                frame[:,:,video_channels] - bkg[vidfile][:,:,video_channels]
                if loss=='exclusive':
                    labels[i - offset] = self.labels_list.index(sound['label'])
                    sounds.append(sound)
                else:
                    target = 0 if sound['label'].startswith(overlapped_prefix) else 1
                    root = sound['label'].removeprefix(overlapped_prefix)
                    labels[i - offset, self.labels_list.index(root)] = target
                    sounds.append([sound])
                    for ioverlap in sound['overlaps']:
                        overlapped_sound = self.data_index[mode][ioverlap]
                        if overlapped_sound['ticks'][1] > sound['ticks'][0] and \
                           overlapped_sound['ticks'][0] < sound['ticks'][1]:
                            target = 0 if overlapped_sound['label'].startswith(overlapped_prefix) else 1
                            root = overlapped_sound['label'].removeprefix(overlapped_prefix)
                            labels[i - offset, self.labels_list.index(root)] = target
                            sounds[-1].append(overlapped_sound)
            if use_audio and use_video:
                q.put([[audio_slice, video_slice], labels, sounds])
            elif use_audio:
                q.put([audio_slice, labels, sounds])
            elif use_video:
                q.put([video_slice, labels, sounds])

    def get_data(self, how_many, offset, model_settings, loss, overlapped_prefix,
                 shiftby_ms, mode, use_audio, use_video, video_findfile):
        """Gather sounds from the data set, applying transformations as needed.

        When the mode is 'training', a random selection of sounds will be returned,
        otherwise the first N clips in the partition will be used. This ensures that
        validation always uses the same sounds, reducing noise in the metrics.

        Args:
          how_many: Desired number of sounds to return. -1 means the entire
            contents of this partition.
          offset: Where to start when fetching deterministically.
          model_settings: Information about the current model being trained.
          time_shift: How much to randomly shift the clips by in time.
          mode: Which partition to use, must be 'training', 'validation', or
            'testing'.
          sess: TensorFlow session that was active when processor was created.

        Returns:
          List of sound data for the transformed sounds, and list of label indexes
        """

        if mode not in queues:
            queues[mode] = Queue(self.queue_size)
            processes[mode] = []
            if mode!='training':
                offsets[mode] = Queue(len(range(0, len(self.data_index[mode]), how_many)))
        if mode!='training' and offset==0:
            # HACK: not guaranteed to be in order
            _offsets = list(range(0, len(self.data_index[mode]), how_many))
            _offsets.reverse()
            for _offset in _offsets:
                offsets[mode].put(_offset)
        if queues[mode].empty() and \
           ((self.max_procs==0 and len(processes[mode])<=cpu_count()) or \
            len(processes[mode])<self.max_procs):
            p = Process(target=self._get_data,
                        args=(queues[mode], offsets[mode] if mode!='training' else None,
                              how_many, offset, model_settings, loss,
                              overlapped_prefix, shiftby_ms,
                              mode, use_audio, use_video, video_findfile),
                        daemon=True)
            p.start()
            processes[mode].append(p)

        return queues[mode].get()
