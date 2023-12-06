#!/usr/bin/env python3

# test that weights_seed works

import sys
import os
import shutil
import glob
from subprocess import run, PIPE, STDOUT, Popen
import time
import math
import asyncio

from libtest import wait_for_job, check_file_exists, get_srcrepobindirs

_, repo_path, bindirs = get_srcrepobindirs()

os.environ['PATH'] = os.pathsep.join([*os.environ['PATH'].split(os.pathsep), *bindirs])
  
sys.path.append(os.path.join(repo_path, "src", "gui"))
import model as M
import view as V
import controller as C

os.makedirs(os.path.join(repo_path, "test", "scratch", "seeds"))
shutil.copy(os.path.join(repo_path, "configuration.py"),
            os.path.join(repo_path, "test", "scratch", "seeds"))

M.init(None, os.path.join(repo_path, "test", "scratch", "seeds", "configuration.py"))
V.init(None)
C.init(None)

os.makedirs(os.path.join(repo_path, "test", "scratch", "seeds", "groundtruth-data", "round1"))
shutil.copy(os.path.join(repo_path, "data", "PS_20130625111709_ch3.wav"),
            os.path.join(repo_path, "test", "scratch", "seeds", "groundtruth-data", "round1"))

run(["hstart", "1,0,1"])

shutil.copy(os.path.join(repo_path, "data", "PS_20130625111709_ch3-annotated-person1.csv"),
            os.path.join(repo_path, "test", "scratch", "seeds", "groundtruth-data", "round1"))

V.context_ms.value = "204.8"
V.shiftby_ms.value = "0.0"
V.optimizer.value = "Adam"
V.learning_rate.value = "0.0002"
V.model_parameters["dropout_kind"].value = "unit"
V.model_parameters["dropout_rate"].value = "50"
V.model_parameters["augment_volume"].value = "1,1"
V.model_parameters["augment_noise"].value = "0,0"
V.model_parameters["normalization"].value = "none"
V.model_parameters["kernel_sizes"].value = "5x5,3"
V.model_parameters["nconvlayers"].value = "2"
V.model_parameters["denselayers"].value = ""
V.model_parameters["nfeatures"].value = "64,64"
V.model_parameters["stride_time"].value = ""
V.model_parameters["stride_freq"].value = ""
V.model_parameters["dilate_time"].value = ""
V.model_parameters["dilate_freq"].value = ""
V.model_parameters["pool_kind"].value = "none"
V.model_parameters["pool_size"].value = ""
V.model_parameters["connection_type"].value = "plain"
V.model_parameters["representation"].value = "mel-cepstrum"
V.model_parameters["window_ms"].value = "6.4"
V.model_parameters["stride_ms"].value = "1.6"
V.model_parameters["mel_dct"].value = "7,7"
V.model_parameters["range_hz"].value = ""
V.groundtruth_folder.value = os.path.join(repo_path, "test", "scratch", "seeds", "groundtruth-data")
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
          "test", "scratch", "seeds", "trained-classifier-bs="+batch_seed+"-ws="+weights_seed)
    asyncio.run(C.train_actuate())

    wait_for_job(M.status_ticker_queue)

    check_file_exists(os.path.join(V.logs_folder.value, "train1.log"))
    check_file_exists(os.path.join(V.logs_folder.value, "train_1r.log"))
    check_file_exists(os.path.join(V.logs_folder.value,
                                   "train_1r", "ckpt-"+V.nsteps.value+".index"))

run(["hstop"], stdout=PIPE, stderr=STDOUT)


import tensorflow as tf
import numpy as np

same_weights = [os.path.join("trained-classifier-bs=1-ws=1", "train_1r", "ckpt-0"),
                os.path.join("trained-classifier-bs=-1-ws=1", "train_1r", "ckpt-0")]
diff_weights = [os.path.join("trained-classifier-bs=1-ws=-1", "train_1r", "ckpt-0"),
                os.path.join("trained-classifier-bs=-1-ws=-1", "train_1r", "ckpt-0")]

model0_var_names = None
for model in [*same_weights, *diff_weights]:
  model_path = os.path.join(repo_path, "test", "scratch", "seeds", model)
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
  model_path = os.path.join(repo_path, "test", "scratch", "seeds", same_weights[0])
  same0_var_value = tf.train.load_variable(model_path, var_name[0])
  for model in same_weights[1:]:
    model_path = os.path.join(repo_path, "test", "scratch", "seeds", model)
    same_var_value = tf.train.load_variable(model_path, var_name[0])
    if not np.all(same0_var_value == same_var_value):
      print("ERROR: "+model+" "+var_name[0]+" not the same")
  for model in diff_weights:
    model_path = os.path.join(repo_path, "test", "scratch", "seeds", model)
    diff_var_value = tf.train.load_variable(model_path, var_name[0])
    if not np.any(same0_var_value != diff_var_value):
      print("ERROR: "+model+" "+var_name[0]+" not different")
