#!/usr/bin/python3

# test that freeze and classify work with different representations,
# downsamplings via strided convolutions, and numbers of windows to predict
# in parallel

# export SINGULARITYENV_SONGEXPLORER_STATE=/tmp
# ${SONGEXPLORER_BIN/-B/-B /tmp:/opt/deepsong/test/scratch -B} test/freeze-classify.py

import sys
import os
import shutil
import glob
from subprocess import run, PIPE, STDOUT, Popen
import time
import math
import asyncio
import filecmp

from lib import wait_for_job, check_file_exists

repo_path = os.path.dirname(sys.path[0])
  
sys.path.append(os.path.join(repo_path, "src/gui"))
import model as M
import view as V
import controller as C

os.makedirs(os.path.join(repo_path, "test/scratch/freeze-classify"))
shutil.copy(os.path.join(repo_path, "configuration.pysh"),
            os.path.join(repo_path, "test/scratch/freeze-classify"))

M.init(None, os.path.join(repo_path, "test/scratch/freeze-classify/configuration.pysh"))
V.init(None)
C.init(None)

os.makedirs(os.path.join(repo_path, "test/scratch/freeze-classify/groundtruth-data/round1"))
shutil.copy(os.path.join(repo_path, "data/PS_20130625111709_ch3.wav"),
            os.path.join(repo_path, "test/scratch/freeze-classify/groundtruth-data/round1"))

run(["hetero", "start", "1", "1", "1"])

shutil.copy(os.path.join(repo_path, "data/PS_20130625111709_ch3-annotated-person1.csv"),
            os.path.join(repo_path, "test/scratch/freeze-classify/groundtruth-data/round1"))

V.context_ms_string.value = "204.8"
V.shiftby_ms_string.value = "0.0"
V.optimizer.value = "adam"
V.learning_rate_string.value = "0.0002"
V.window_ms_string.value = "6.4"
V.stride_ms_string.value = "1.6"
V.mel_dct_string.value = "7,7"
V.model_parameters["dropout"].value = "0.5"
V.model_parameters["kernel_sizes"].value = "5,3,128"
V.model_parameters["nlayers"].value = "2"
V.model_parameters["nfeatures"].value = "64,64,64"
V.model_parameters["dilate_after_layer"].value = "65535"
V.model_parameters["connection_type"].value = "plain"
V.groundtruth_folder.value = os.path.join(repo_path,
                                          "test/scratch/freeze-classify/groundtruth-data")
V.wantedwords_string.value = "mel-pulse,mel-sine,ambient"
V.labeltypes_string.value = "annotated"
V.nsteps_string.value = "100"
V.restore_from_string.value = ""
V.save_and_validate_period_string.value = "10"
V.validate_percentage_string.value = "40"
V.mini_batch_string.value = "32"
V.testing_files = ""
V.batch_seed_string.value = "1"
V.weights_seed_string.value = "1"
V.replicates_string.value = "1"

for representation in ["waveform", "spectrogram", "mel-cepstrum"]:
  V.representation.value = representation
  for stride_after_layer in ["0", "65535"]:
    V.model_parameters["stride_after_layer"].value = stride_after_layer
    V.logs_folder.value = os.path.join(repo_path,
          "test/scratch/freeze-classify",
          "trained-classifier-r="+representation+"-s="+stride_after_layer)
    asyncio.run(C.train_actuate())

    wait_for_job(M.status_ticker_queue)

    check_file_exists(os.path.join(V.logs_folder.value, "train1.log"))
    check_file_exists(os.path.join(V.logs_folder.value, "train_1r.log"))
    check_file_exists(os.path.join(V.logs_folder.value,
                                   "train_1r", "ckpt-"+V.nsteps_string.value+".index"))

    V.model_file.value = os.path.join(V.logs_folder.value,
                                      "train_"+V.replicates_string.value+"r",
                                      "ckpt-"+V.nsteps_string.value+".meta")
    V.wavtfcsvfiles_string.value = os.path.join(repo_path,
          "test/scratch/freeze-classify/groundtruth-data/round1/PS_20130625111709_ch3.wav")
    V.prevalences_string.value = ""

    for nwindows in ["1", "9"]:
      asyncio.run(C.freeze_actuate())
      wait_for_job(M.status_ticker_queue)

      check_file_exists(os.path.join(V.logs_folder.value, "train_1r",
                                     "freeze.ckpt-"+V.nsteps_string.value+".log"))
      check_file_exists(os.path.join(V.logs_folder.value, "train_1r",
                                     "frozen-graph.ckpt-"+V.nsteps_string.value+".pb"))

      asyncio.run(C.classify_actuate())

      wait_for_job(M.status_ticker_queue)

      outpath = os.path.join(repo_path, 
                             "test/scratch/freeze-classify",
                             "trained-classifier-r="+representation+"-s="+stride_after_layer,
                             nwindows)
      os.makedirs(outpath)

      wavpath_noext = V.wavtfcsvfiles_string.value[:-4]
      check_file_exists(wavpath_noext+".tf")
      check_file_exists(wavpath_noext+"-classify1.log")
      check_file_exists(wavpath_noext+"-classify2.log")
      for word in V.wantedwords_string.value.split(','):
        check_file_exists(wavpath_noext+"-"+word+".wav")
        shutil.copy(wavpath_noext+"-"+word+".wav", outpath)

    outpath = os.path.join(repo_path, 
                           "test/scratch/freeze-classify",
                           "trained-classifier-r="+representation+"-s="+stride_after_layer)
    for word in V.wantedwords_string.value.split(','):
      outpath1 = os.path.join(outpath, "1", wavpath_noext+"-"+word+".wav")
      outpath9 = os.path.join(outpath, "9", wavpath_noext+"-"+word+".wav")
      if not filecmp.cmp(outpath1, outpath9, shallow=False):
        print("ERROR: "+outpath1+" and "+outpath9+" are different")

run(["hetero", "stop"], stdout=PIPE, stderr=STDOUT)
