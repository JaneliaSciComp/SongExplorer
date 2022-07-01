#!/usr/bin/env python3

# train several networks on different subsets of the annotations

# xvalidate.sh <context-ms> <shiftby-ms> <optimizer> <learning-rate> <video-findfile> <video-bkg-frames> <data-loader-queuesize> <data-loader-maxprocs> <model-architecture> <model-parameters-json> <logdir> <path-to-groundtruth> <label1>,<label2>,...,<labelN> <kinds-to-use> <nsteps> <restore-from> <save-and-validate-period> <mini-batch> <testing-files> <audio-tic-rate> <audio-nchannels> <video-frame-rate> <video-frame-width> <video-frame-height> <video_channels> <batch-seed> <weights-seed> <deterministic> <kfold> <ifolds>

# e.g.
# $SONGEXPLORER_BIN xvalidate.py 204.8 0.0 Adam 0.0002 same-basename 1000 0 1 convolutional '{"representation":"waveform", "window_ms":6.4, "stride_ms":1.6, "mel_dct":"7,7", "dropout":0.5, "kernel_sizes":5,128", last_conv_width":130, "nfeatures":"256,256", "dilate_after_layer":65535, "stride_after_layer":65535, "connection_type":"plain"}' `pwd`/cross-validate `pwd`/groundtruth-data mel-pulse,mel-sine,ambient,other annotated 50 '' 10 32 "" 5000 1 0 0 0 0 -1 -1 0 8 1,2

import os
import sys
from subprocess import run, PIPE, STDOUT
import asyncio

from datetime import datetime
import socket

print(str(datetime.now())+": start time")
repodir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
with open(os.path.join(repodir, "VERSION.txt"), 'r') as fid:
  print('SongExplorer version = '+fid.read().strip().replace('\n',', '))
print("hostname = "+socket.gethostname())
print("CUDA_VISIBLE_DEVICES = "+os.environ.get('CUDA_VISIBLE_DEVICES',''))
p = run('which nvidia-smi && nvidia-smi', shell=True, stdout=PIPE, stderr=STDOUT)
print(p.stdout.decode('ascii').rstrip())

try:

  _, context_ms, shiftby_ms, optimizer, learning_rate, video_findfile, video_bkg_frames, data_loader_queuesize, data_loader_maxprocs, architecture, model_parameters, logdir, data_dir, labels_touse, kinds_touse, nsteps, restore_from, save_and_validate_period, mini_batch, testing_files, audio_tic_rate, audio_nchannels, video_frame_rate, video_frame_width, video_frame_height, video_channels, batch_seed, weights_seed, deterministic, kfold, ifolds = sys.argv[:31]

  print('context_ms: '+context_ms)
  print('shiftby_ms: '+shiftby_ms)
  print('optimizer: '+optimizer)
  print('learning_rate: '+learning_rate)
  print('video_findfile: '+video_findfile)
  print('video_bkg_frames: '+video_bkg_frames)
  print('data_loader_queuesize: '+data_loader_queuesize)
  print('data_loader_maxprocs: '+data_loader_maxprocs)
  print('architecture: '+architecture)
  print('model_parameters: '+model_parameters)
  print('logdir: '+logdir)
  print('data_dir: '+data_dir)
  print('labels_touse: '+labels_touse)
  print('kinds_touse: '+kinds_touse)
  print('nsteps: '+nsteps)
  print('restore_from: '+restore_from)
  print('save_and_validate_period: '+save_and_validate_period)
  print('mini_batch: '+mini_batch)
  print('testing_files: '+testing_files)
  print('audio_tic_rate: '+audio_tic_rate)
  print('audio_nchannels: '+audio_nchannels)
  print('video_frame_rate: '+video_frame_rate)
  print('video_frame_width: '+video_frame_width)
  print('video_frame_height: '+video_frame_height)
  print('video_channels: '+video_channels)
  print('batch_seed: '+batch_seed)
  print('weights_seed: '+weights_seed)
  print('deterministic: '+deterministic)
  print('kfold: '+kfold)
  print('ifolds: '+ifolds)

  if restore_from:
    mode='a'
    start_checkpoint=os.path.join(logdir, "xvalidate_MODEL", "ckpt-"+restore_from)
  else:
    mode='w'
    start_checkpoint=''

  async def redirect(cmd):
    with open(cmd[-1], 'a') as fid:
      proc = await asyncio.create_subprocess_exec(*cmd[:-1],
                                                  stderr=asyncio.subprocess.PIPE,
                                                  stdout=fid)
      await proc.communicate()

  async def main():
    cmds = []
    for ifold in ifolds.split(','):
      model=ifold+'k'
      kpercent=100/int(kfold)
      koffset=kpercent * (int(ifold) - 1)
      expr=["loop.py",
            "--context_ms="+context_ms,
            "--shiftby_ms="+shiftby_ms,
            "--optimizer="+optimizer,
            "--learning_rate="+learning_rate,
            "--video_findfile="+video_findfile,
            "--video_bkg_frames="+video_bkg_frames,
            "--data_loader_queuesize="+data_loader_queuesize,
            "--data_loader_maxprocs="+data_loader_maxprocs,
            "--model_architecture="+architecture,
            "--model_parameters="+model_parameters,
            "--data_dir="+data_dir,
            "--labels_touse="+labels_touse,
            "--kinds_touse="+kinds_touse,
            "--how_many_training_steps="+nsteps,
            "--start_checkpoint="+start_checkpoint.replace("MODEL",model),
            "--save_step_period="+save_and_validate_period,
            "--validate_step_period="+save_and_validate_period,
            "--batch_size="+mini_batch,
            "--testing_files="+testing_files,
            "--audio_tic_rate="+audio_tic_rate,
            "--audio_nchannels="+audio_nchannels,
            "--video_frame_rate="+video_frame_rate,
            "--video_frame_width="+video_frame_width,
            "--video_frame_height="+video_frame_height,
            "--video_channels="+video_channels,
            "--random_seed_batch="+batch_seed,
            "--random_seed_weights="+weights_seed,
            "--deterministic="+deterministic,
            "--validation_percentage="+str(kpercent),
            "--validation_offset_percentage="+str(koffset),
            "--train_dir="+os.path.join(logdir,"xvalidate_"+model),
            "--summaries_dir="+os.path.join(logdir,"summaries_"+model)]

            #"--subsample_label=mel-pulse",
            #"--subsample_skip=1",

      cmds.append(expr+[os.path.join(logdir,"xvalidate_"+model+".log")])
      with open(cmds[-1][-1], mode) as fid:
        fid.write(' '.join(cmds[-1][:-1])+'\n')

    await asyncio.gather(*[redirect(x) for x in cmds])

  asyncio.run(main())

except Exception as e:
  print(e)

finally:
  os.sync()
  print(str(datetime.now())+": finish time")
