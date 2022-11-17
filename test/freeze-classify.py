#!/usr/bin/env python3

# test that freeze and classify work with different representations,
# downsamplings via strided convolutions, and numbers of outputs tics
# to predict in parallel

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
import scipy.io.wavfile as spiowav

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

run(["hetero", "start", "1", "0", "1"])

shutil.copy(os.path.join(repo_path, "data/PS_20130625111709_ch3-annotated-person1.csv"),
            os.path.join(repo_path, "test/scratch/freeze-classify/groundtruth-data/round1"))

V.context_ms.value = "204.8"
V.shiftby_ms.value = "0.0"
V.optimizer.value = "Adam"
V.learning_rate.value = "0.0002"
V.model_parameters["dropout_kind"].value = "unit"
V.model_parameters["dropout_rate"].value = "50"
V.model_parameters["augment_volume"].value = "1,1"
V.model_parameters["augment_noise"].value = "0,0"
V.model_parameters["normalization"].value = "none"
V.model_parameters["kernel_sizes"].value = "3x3,32"
V.model_parameters["nconvlayers"].value = "2"
V.model_parameters["denselayers"].value = ""
V.model_parameters["nfeatures"].value = "16,16"
V.model_parameters["stride_freq"].value = ""
V.model_parameters["dilate_time"].value = ""
V.model_parameters["dilate_freq"].value = ""
V.model_parameters["pool_kind"].value = "none"
V.model_parameters["pool_size"].value = ""
V.model_parameters["connection_type"].value = "plain"
V.model_parameters["window_ms"].value = "3.2"
V.model_parameters["stride_ms"].value = "0.8"
V.model_parameters["mel_dct"].value = "3,3"
V.model_parameters["range_hz"].value = ""
V.groundtruth_folder.value = os.path.join(repo_path,
                                          "test/scratch/freeze-classify/groundtruth-data")
V.labels_touse.value = "mel-pulse,mel-sine,ambient"
V.kinds_touse.value = "annotated"
V.nsteps.value = "100"
V.restore_from.value = ""
V.save_and_validate_period.value = "10"
V.validate_percentage.value = "40"
V.mini_batch.value = "32"
V.test_files.value = ""
V.batch_seed.value = "1"
V.weights_seed.value = "1"
V.nreplicates.value = "1"

for representation in ["waveform", "spectrogram", "mel-cepstrum"]:
  V.model_parameters["representation"].value = representation
  for stride_time in ["2", ""]:
    V.model_parameters["stride_time"].value = stride_time
    V.logs_folder.value = os.path.join(repo_path,
                                       "test/scratch/freeze-classify",
                                       "trained-classifier-r="+representation+
                                       "-s="+stride_time)
    asyncio.run(C.train_actuate())

    wait_for_job(M.status_ticker_queue)

    check_file_exists(os.path.join(V.logs_folder.value, "train1.log"))
    check_file_exists(os.path.join(V.logs_folder.value, "train_1r.log"))
    check_file_exists(os.path.join(V.logs_folder.value,
                                   "train_1r", "ckpt-"+V.nsteps.value+".index"))

    V.model_file.value = os.path.join(V.logs_folder.value,
                                      "train_"+V.nreplicates.value+"r",
                                      "ckpt-"+V.nsteps.value+".index")
    V.wavcsv_files.value = os.path.join(repo_path,
          "test/scratch/freeze-classify/groundtruth-data/round1/PS_20130625111709_ch3.wav")
    V.prevalences.value = ""

    for parallelize in ["64", "16384"]:
      M.classify_parallelize=int(parallelize)
      asyncio.run(C.freeze_actuate())
      wait_for_job(M.status_ticker_queue)

      check_file_exists(os.path.join(V.logs_folder.value, "train_1r",
                                     "freeze.ckpt-"+V.nsteps.value+".log"))
      check_file_exists(os.path.join(V.logs_folder.value, "train_1r",
                                     "frozen-graph.ckpt-"+V.nsteps.value+".pb",
                                     "saved_model.pb"))

      asyncio.run(C.classify_actuate())

      wait_for_job(M.status_ticker_queue)

      outpath = os.path.join(repo_path, 
                             "test/scratch/freeze-classify",
                             "trained-classifier-r="+representation+
                             "-s="+stride_time,
                             parallelize)
      os.makedirs(outpath)

      wavpath_noext = V.wavcsv_files.value[:-4]
      check_file_exists(wavpath_noext+"-classify.log")
      shutil.move(wavpath_noext+"-classify.log", outpath)
      for label in V.labels_touse.value.split(','):
        check_file_exists(wavpath_noext+"-"+label+".wav")
        shutil.move(wavpath_noext+"-"+label+".wav", outpath)

    outpath = os.path.join(repo_path, 
                           "test/scratch/freeze-classify",
                           "trained-classifier-r="+representation+
                           "-s="+stride_time)
    for label in V.labels_touse.value.split(','):
      wavbase_noext = os.path.basename(wavpath_noext)
      outpath64 = os.path.join(outpath, "64", wavbase_noext+"-"+label+".wav")
      outpath16384 = os.path.join(outpath, "16384", wavbase_noext+"-"+label+".wav")
      _, wavs64 = spiowav.read(outpath64)
      _, wavs16384 = spiowav.read(outpath16384)
      if len(wavs64) != len(wavs16384):
        print("ERROR: "+outpath64+" and "+outpath16384+" are of different lengths")
      elif any(wavs64 != wavs16384):
        print("ERROR: "+outpath64+" and "+outpath16384+
              " differ by at most "+str(max(abs(wavs64-wavs16384)))+" part(s) in 2^16"+
              " for "+str(sum(wavs64!=wavs16384))+" samples")

run(["hetero", "stop"], stdout=PIPE, stderr=STDOUT)
