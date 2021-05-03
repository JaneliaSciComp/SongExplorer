#!/usr/bin/python3

# test that weights_seed works

# export SINGULARITYENV_SONGEXPLORER_STATE=/tmp
# ${SONGEXPLORER_BIN/-B/-B /tmp:/opt/songexplorer/test/scratch -B} test/seeds.py

import sys
import os
import shutil
import glob
from subprocess import run, PIPE, STDOUT, Popen
import time
import math
import asyncio

from lib import wait_for_job, check_file_exists

repo_path = os.path.dirname(sys.path[0])
  
sys.path.append(os.path.join(repo_path, "src/gui"))
import model as M
import view as V
import controller as C

os.makedirs(os.path.join(repo_path, "test/scratch/seeds"))
shutil.copy(os.path.join(repo_path, "configuration.pysh"),
            os.path.join(repo_path, "test/scratch/seeds"))

M.init(None, os.path.join(repo_path, "test/scratch/seeds/configuration.pysh"))
V.init(None)
C.init(None)

os.makedirs(os.path.join(repo_path, "test/scratch/seeds/groundtruth-data/round1"))
shutil.copy(os.path.join(repo_path, "data/PS_20130625111709_ch3.wav"),
            os.path.join(repo_path, "test/scratch/seeds/groundtruth-data/round1"))

run(["hetero", "start", "1", "1", "1"])

shutil.copy(os.path.join(repo_path, "data/PS_20130625111709_ch3-annotated-person1.csv"),
            os.path.join(repo_path, "test/scratch/seeds/groundtruth-data/round1"))

V.context_ms.value = "204.8"
V.shiftby_ms.value = "0.0"
V.optimizer.value = "Adam"
V.learning_rate.value = "0.0002"
V.representation.value = "mel-cepstrum"
V.window_ms.value = "6.4"
V.stride_ms.value = "1.6"
V.mel_dct.value = "7,7"
V.model_parameters["dropout"].value = "0.5"
V.model_parameters["kernel_sizes"].value = "5,3,3"
V.model_parameters["nlayers"].value = "2"
V.model_parameters["nfeatures"].value = "64,64,64"
V.model_parameters["dilate_after_layer"].value = "65535"
V.model_parameters["stride_after_layer"].value = "65535"
V.model_parameters["connection_type"].value = "plain"
V.groundtruth_folder.value = os.path.join(repo_path, "test/scratch/seeds/groundtruth-data")
V.labels_touse.value = "mel-pulse,mel-sine,ambient"
V.kinds_touse.value = "annotated"
V.nsteps.value = "100"
V.restore_from.value = ""
V.save_and_validate_period.value = "10"
V.validate_percentage.value = "40"
V.mini_batch.value = "32"
V.test_files.value = ""
V.nreplicates.value = "1"

for batch_seed in ["1", "-1"]:
  V.batch_seed.value = batch_seed
  for weights_seed in ["1", "-1"]:
    V.weights_seed.value = weights_seed
    V.logs_folder.value = os.path.join(repo_path,
          "test/scratch/seeds/trained-classifier-bs="+batch_seed+"-ws="+weights_seed)
    asyncio.run(C.train_actuate())

    wait_for_job(M.status_ticker_queue)

    check_file_exists(os.path.join(V.logs_folder.value, "train1.log"))
    check_file_exists(os.path.join(V.logs_folder.value, "train_1r.log"))
    check_file_exists(os.path.join(V.logs_folder.value,
                                   "train_1r", "ckpt-"+V.nsteps.value+".index"))

run(["hetero", "stop"], stdout=PIPE, stderr=STDOUT)


import tensorflow as tf
import numpy as np

same_weights = ["trained-classifier-bs=1-ws=1/train_1r/ckpt-0",
                "trained-classifier-bs=-1-ws=1/train_1r/ckpt-0"]
diff_weights = ["trained-classifier-bs=1-ws=-1/train_1r/ckpt-0",
                "trained-classifier-bs=-1-ws=-1/train_1r/ckpt-0"]

model0_var_names = None
for model in [*same_weights, *diff_weights]:
  model_path = os.path.join(repo_path, "test/scratch/seeds", model)
  model_var_names = tf.train.list_variables(model_path)
  if not model0_var_names:
    model0_var_names = model_var_names
    continue
  if model0_var_names != model_var_names:
    print("ERROR: var list not the same")
    print(model_path+" vars: "+str(model_var_names))

for var_name in model0_var_names:
  if len(var_name[1])<2:  # biases
    continue
  model_path = os.path.join(repo_path, "test/scratch/seeds", same_weights[0])
  same0_var_value = tf.train.load_variable(model_path, var_name[0])
  for model in same_weights[1:]:
    model_path = os.path.join(repo_path, "test/scratch/seeds", model)
    same_var_value = tf.train.load_variable(model_path, var_name[0])
    if not np.all(same0_var_value == same_var_value):
      print("ERROR: "+model+" "+var_name[0]+" not the same")
  for model in diff_weights:
    model_path = os.path.join(repo_path, "test/scratch/seeds", model)
    diff_var_value = tf.train.load_variable(model_path, var_name[0])
    if not np.any(same0_var_value != diff_var_value):
      print("ERROR: "+model+" "+var_name[0]+" not different")
