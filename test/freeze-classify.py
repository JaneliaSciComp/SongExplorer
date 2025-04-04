#!/usr/bin/env python

# test that freeze and classify work with different representations,
# downsamplings via strided convolutions, and numbers of outputs tics
# to predict in parallel

import sys
import os
import shutil
import glob
from subprocess import run, PIPE, STDOUT, Popen
import time
import math
import asyncio
import scipy.io.wavfile as spiowav

from libtest import wait_for_job, check_file_exists, get_srcrepobindirs

_, repo_path, bindirs = get_srcrepobindirs()

os.environ['PATH'] = os.pathsep.join([*bindirs, *os.environ['PATH'].split(os.pathsep)])
  
sys.path.append(os.path.join(repo_path, "src", "gui"))
import model as M
import view as V
import controller as C

os.makedirs(os.path.join(repo_path, "test", "scratch", "freeze-classify"))
shutil.copy(os.path.join(repo_path, "configuration.py"),
            os.path.join(repo_path, "test", "scratch", "freeze-classify"))

M.init(None, os.path.join(repo_path, "test", "scratch", "freeze-classify", "configuration.py"), True)
V.init(None)
C.init(None)

os.makedirs(os.path.join(repo_path, "test", "scratch", "freeze-classify", "groundtruth-data", "round1"))
shutil.copy(os.path.join(repo_path, "data", "PS_20130625111709_ch3.wav"),
            os.path.join(repo_path, "test", "scratch", "freeze-classify", "groundtruth-data", "round1"))

run(["hstart", "1,0,1"])

shutil.copy(os.path.join(repo_path, "data", "PS_20130625111709_ch3.wav-annotated-person1.csv"),
            os.path.join(repo_path, "test", "scratch", "freeze-classify", "groundtruth-data", "round1"))

V.context.value = "204.8"
V.shiftby.value = "0.0"
V.optimizer.value = "Adam"
V.learning_rate.value = "0.0002"
V.model_parameters["dropout"].value = "50"
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
V.model_parameters["window"].value = "3.2"
V.model_parameters["stride"].value = "0.8"
V.model_parameters["mel_dct"].value = "3,3"
V.model_parameters["range"].value = ""
V.groundtruth_folder.value = os.path.join(repo_path,
                                          "test", "scratch", "freeze-classify", "groundtruth-data")
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
                                       "test", "scratch", "freeze-classify",
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
          "test", "scratch", "freeze-classify", "groundtruth-data", "round1", "PS_20130625111709_ch3.wav")
    V.prevalences.value = ""

    for parallelize in ["64", "16384"]:
      V.parallelize.value=str(parallelize)
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
                             "test", "scratch", "freeze-classify",
                             "trained-classifier-r="+representation+
                             "-s="+stride_time,
                             parallelize)
      os.makedirs(outpath)

      wavpath = V.wavcsv_files.value
      check_file_exists(wavpath+"-classify.log")
      shutil.move(wavpath+"-classify.log", outpath)
      for label in V.labels_touse.value.split(','):
        check_file_exists(wavpath+"-"+label+".wav")
        shutil.move(wavpath+"-"+label+".wav", outpath)

    outpath = os.path.join(repo_path, 
                           "test", "scratch", "freeze-classify",
                           "trained-classifier-r="+representation+
                           "-s="+stride_time)
    for label in V.labels_touse.value.split(','):
      wavbase = os.path.basename(wavpath)
      outpath64 = os.path.join(outpath, "64", wavbase+"-"+label+".wav")
      outpath16384 = os.path.join(outpath, "16384", wavbase+"-"+label+".wav")
      _, wavs64 = spiowav.read(outpath64)
      _, wavs16384 = spiowav.read(outpath16384)
      if len(wavs64) != len(wavs16384):
        print("ERROR: "+outpath64+" and "+outpath16384+" are of different lengths")
      elif any(wavs64 != wavs16384):
        print("ERROR: "+outpath64+" and "+outpath16384+
              " differ by at most "+str(max(abs(wavs64-wavs16384)))+" part(s) in 2^16"+
              " for "+str(sum(wavs64!=wavs16384))+" samples")

run(["hstop"], stdout=PIPE, stderr=STDOUT)
