#!/usr/bin/env python3

# test that shift_by works

import sys
import os
import shutil
from subprocess import run
import numpy as np
import platform

from libtest import get_srcrepobindirs

srcdir, repo_path, bindirs = get_srcrepobindirs()

os.environ['PATH'] = os.pathsep.join([*os.environ['PATH'].split(os.pathsep), *bindirs])

os.makedirs(os.path.join(repo_path, "test", "scratch", "shiftby"))
shutil.copy(os.path.join(repo_path, "configuration.py"),
            os.path.join(repo_path, "test", "scratch", "shiftby"))

os.makedirs(os.path.join(repo_path, "test", "scratch", "shiftby", "groundtruth-data", "round1"))
shutil.copy(os.path.join(repo_path, "data", "PS_20130625111709_ch3.wav"), \
            os.path.join(repo_path, "test", "scratch", "shiftby", "groundtruth-data", "round1"))
shutil.copy(os.path.join(repo_path, "data", "PS_20130625111709_ch3-annotated-person1.csv"), \
            os.path.join(repo_path, "test", "scratch", "shiftby", "groundtruth-data", "round1"))

parameters = [
    "--context=204.8",
    "--optimizer=Adam",
    "--learning_rate=0.000001",
    "--audio_read_plugin=load-wav",
    "--audio_read_plugin_kwargs={}",
    "--video_read_plugin=load-avi-mp4-mov",
    "--video_read_plugin_kwargs={}",
    "--video_findfile=same-basename",
    "--video_bkg_frames=0",
    "--data_loader_queuesize=1",
    "--data_loader_maxprocs=0",
    "--model_architecture=convolutional",
    '--model_parameters={"representation": "waveform", \
                         "window": "3.2", \
                         "stride": "0.8", \
                         "range": "", \
                         "mel_dct": "3,3", \
                         "connection_type": "plain", \
                         "nconvlayers": "2", \
                         "kernel_sizes": "3x3,32", \
                         "nfeatures": "8,8", \
                         "dropout_kind": "unit", \
                         "dropout_rate": "50", \
                         "normalization": "none", \
                         "stride_time": "2", \
                         "stride_freq": "", \
                         "dilate_time": "", \
                         "dilate_freq": "", \
                         "pool_kind": "none", \
                         "pool_size": "", \
                         "denselayers": "", \
                         "augment_volume": "1,1", \
                         "augment_noise": "0,0"}',
    "--data_dir="+os.path.join(repo_path, "test", "scratch", "shiftby", "groundtruth-data"),
    "--labels_touse=mel-pulse,mel-sine,ambient",
    "--kinds_touse=annotated",
    "--nsteps=10",
    "--restore_from=",
    "--save_and_validate_period=5",
    "--validation_percentage=20",
    "--mini_batch=32",
    "--testing_files=",
    "--time_units=ms",
    "--freq_units=Hz",
    "--time_scale=0.001",
    "--freq_scale=1",
    "--audio_tic_rate=2500",
    "--audio_nchannels=1",
    "--video_frame_rate=0",
    "--video_frame_width=0",
    "--video_frame_height=0",
    "--video_channels=0",
    "--batch_seed=1",
    "--weights_seed=1",
    "--deterministic=0",
    "--igpu=",
    "--ireplicates=1",
    "--save_fingerprints=1"]  # not hoised to GUI

shiftby = 0.0
logdir = os.path.join(repo_path, "test", "scratch", "shiftby", "shiftby-"+str(shiftby))

os.makedirs(logdir)
cmd = ["python", os.path.join(srcdir, "train"),
        *parameters, "--shiftby="+str(shiftby), "--logdir="+logdir]
with open(os.path.join(logdir, "train1.log"), 'w') as f:
    f.write(str(cmd))
p = run(cmd, capture_output=True)
with open(os.path.join(logdir, "train1.log"), 'a') as f:
    f.write(p.stdout.decode('ascii'))
    f.write(p.stderr.decode('ascii'))

shiftby = 51.2
logdir = os.path.join(repo_path, "test", "scratch", "shiftby", "shiftby-"+str(shiftby))

os.makedirs(logdir)
cmd = ["python", os.path.join(srcdir, "train"),
        *parameters, "--shiftby="+str(shiftby), "--logdir="+logdir]
with open(os.path.join(logdir, "train1.log"), 'w') as f:
    f.write(str(cmd))
p = run(cmd, capture_output=True)
with open(os.path.join(logdir, "train1.log"), 'a') as f:
    f.write(p.stdout.decode('ascii'))
    f.write(p.stderr.decode('ascii'))


fingerprints0 = np.load(os.path.join(repo_path,
                                     "test", "scratch", "shiftby", "shiftby-0.0", "train_1r", "fingerprints.validation.ckpt-10.npz"),
                        allow_pickle=True)
fingerprints512 = np.load(os.path.join(repo_path,
                                       "test", "scratch", "shiftby", "shiftby-51.2", "train_1r", "fingerprints.validation.ckpt-10.npz"),
                          allow_pickle=True)

shiftby = round(51.2/1000*2500)
for isound in range(np.shape(fingerprints0['arr_0'])[0]):
  if not all(fingerprints0['arr_0'][isound,:-shiftby,0] == fingerprints512['arr_0'][isound,shiftby:,0]):
    print("ERROR: traces are not shifted properly")
  if all(fingerprints512['arr_0'][isound,:-shiftby,0]==0):
    print("ERROR: shifted traces appear to be zero padded")
