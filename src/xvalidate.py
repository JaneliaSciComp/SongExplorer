#!/usr/bin/python3

# train several networks on different subsets of the annotations

# xvalidate.sh <context-ms> <shiftby-ms> <optimizer> <learning-rate> <model-architecture> <model-parameters-json> <logdir> <path-to-groundtruth> <label1>,<label2>,...,<labelN> <kinds-to-use> <nsteps> <restore-from> <save-and-validate-period> <mini-batch> <testing-files> <audio-tic-rate> <audio-nchannels> <batch-seed> <weights-seed> <kfold> <ifolds>

# e.g.
# $SONGEXPLORER_BIN xvalidate.sh 204.8 0.0 Adam 0.0002 convolutional '{"representation":"waveform", "window_ms":6.4, "stride_ms":1.6, "mel_dct":"7,7", "dropout":0.5, "kernel_sizes":5,128", last_conv_width":130, "nfeatures":"256,256", "dilate_after_layer":65535, "stride_after_layer":65535, "connection_type":"plain"}' `pwd`/cross-validate `pwd`/groundtruth-data mel-pulse,mel-sine,ambient,other annotated 50 '' 10 32 "" 5000 1 -1 -1 8 1,2

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

  _, context_ms, shiftby_ms, optimizer, learning_rate, architecture, model_parameters, logdir, data_dir, labels_touse, kinds_touse, nsteps, restore_from, save_and_validate_period, mini_batch, testing_files, audio_tic_rate, audio_nchannels, batch_seed, weights_seed, kfold, ifolds = sys.argv[:22]

  print('context_ms: '+context_ms)
  print('shiftby_ms: '+shiftby_ms)
  print('optimizer: '+optimizer)
  print('learning_rate: '+learning_rate)
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
  print('batch_seed: '+batch_seed)
  print('weights_seed: '+weights_seed)
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
      run(cmd[:-1], stderr=PIPE, stdout=fid)

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
            "--nchannels="+audio_nchannels,
            "--random_seed_batch="+batch_seed,
            "--random_seed_weights="+weights_seed,
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
