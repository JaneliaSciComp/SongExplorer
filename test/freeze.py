#!/usr/bin/python3

# test that freeze works with strides and different representations

# export SINGULARITYENV_DEEPSONG_STATE=/tmp
# ${DEEPSONG_BIN/-B/-B /tmp:/opt/deepsong/test/scratch -B} test/freeze.py

import sys
import os
import shutil
import glob
from subprocess import run, PIPE, STDOUT, Popen
import time
import math

from lib import wait_for_job, check_file_exists

repo_path = os.path.dirname(sys.path[0])
  
sys.path.append(os.path.join(repo_path, "src/gui"))
import model as M
import view as V
import controller as C

os.makedirs(os.path.join(repo_path, "test/scratch/freeze"))
shutil.copy(os.path.join(repo_path, "configuration.pysh"),
            os.path.join(repo_path, "test/scratch/freeze"))

M.init(None, os.path.join(repo_path, "test/scratch/freeze/configuration.pysh"))
V.init(None)
C.init(None)

os.makedirs(os.path.join(repo_path, "test/scratch/freeze/groundtruth-data/round1"))
shutil.copy(os.path.join(repo_path, "data/PS_20130625111709_ch3.wav"),
            os.path.join(repo_path, "test/scratch/freeze/groundtruth-data/round1"))

run(["hetero", "start", "1", "1", "1"])

shutil.copy(os.path.join(repo_path, "data/PS_20130625111709_ch3-annotated-person1.csv"),
            os.path.join(repo_path, "test/scratch/freeze/groundtruth-data/round1"))

V.context_ms_string.value = "204.8"
V.shiftby_ms_string.value = "0.0"
V.window_ms_string.value = "6.4"
V.mel_dct_string.value = "7,7"
V.stride_ms_string.value = "1.6"
V.dropout_string.value = "0.5"
V.optimizer.value = "adam"
V.learning_rate_string.value = "0.0002"
V.kernel_sizes_string.value = "5,3,128"
V.last_conv_width_string.value = "130"
V.nfeatures_string.value = "64,64,64"
V.dilate_after_layer_string.value = "65535"
V.connection_type.value = "plain"
V.groundtruth_folder.value = os.path.join(repo_path, "test/scratch/freeze/groundtruth-data")
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
    V.stride_after_layer_string.value = stride_after_layer
    V.logs_folder.value = os.path.join(repo_path,
          "test/scratch/freeze/trained-classifier-r="+representation+"-s="+stride_after_layer)
    C.train_actuate()

    wait_for_job(M.status_ticker_queue)

    check_file_exists(os.path.join(V.logs_folder.value, "train1.log"))
    check_file_exists(os.path.join(V.logs_folder.value, "train_1r.log"))
    check_file_exists(os.path.join(V.logs_folder.value,
                                   "train_1r", "vgg.ckpt-"+V.nsteps_string.value+".index"))

    V.model_file.value = os.path.join(V.logs_folder.value,
                                      "train_"+V.replicates_string.value+"r",
                                      "vgg.ckpt-"+V.nsteps_string.value+".meta")
    C.freeze_actuate()
    wait_for_job(M.status_ticker_queue)

    check_file_exists(os.path.join(V.logs_folder.value, "train_1r",
                                   "freeze.ckpt-"+V.nsteps_string.value+".log"))
    check_file_exists(os.path.join(V.logs_folder.value, "train_1r",
                                   "frozen-graph.ckpt-"+V.nsteps_string.value+".pb"))


run(["hetero", "stop"], stdout=PIPE, stderr=STDOUT)
