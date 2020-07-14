# test that weights_seed works

import sys
import os
import shutil
import glob
from subprocess import run, PIPE, STDOUT, Popen
import time
import math

sys.path.append(os.path.join(os.environ["DEEPSONG_HOME"], "src", "gui"))
import model as M
import view as V
import controller as C

sys.path.append(os.path.join(os.environ["DEEPSONG_HOME"], "test"))
import lib

os.makedirs("scratch/seeds")
shutil.copy("/opt/deepsong/configuration.pysh", "scratch/seeds")

M.init("scratch/seeds/configuration.pysh")
V.init(None)
C.init(None)

os.makedirs("scratch/seeds/groundtruth-data/round1")
shutil.copy("/opt/deepsong/data/PS_20130625111709_ch3.wav",
            "scratch/seeds/groundtruth-data/round1")

run(["hetero", "start",
     str(M.local_ncpu_cores), str(M.local_ngpu_cards), str(M.local_ngigabytes_memory)])

shutil.copy("/opt/deepsong/data/PS_20130625111709_ch3-annotated-person1.csv", \
      "scratch/seeds/groundtruth-data/round1")

V.context_ms_string.value = "204.8"
V.shiftby_ms_string.value = "0.0"
V.representation.value = "mel-cepstrum"
V.window_ms_string.value = "6.4"
V.mel_dct_string.value = "7,7"
V.stride_ms_string.value = "1.6"
V.dropout_string.value = "0.5"
V.optimizer.value = "adam"
V.learning_rate_string.value = "0.0002"
V.kernel_sizes_string.value = "5,3,3"
V.last_conv_width_string.value = "130"
V.nfeatures_string.value = "64,64,64"
V.dilate_after_layer_string.value = "65535"
V.stride_after_layer_string.value = "65535"
V.connection_type.value = "plain"
V.groundtruth_folder.value = "scratch/seeds/groundtruth-data"
V.wantedwords_string.value = "pulse,sine,ambient"
V.labeltypes_string.value = "annotated"
V.nsteps_string.value = "100"
V.restore_from_string.value = ""
V.save_and_validate_period_string.value = "10"
V.validate_percentage_string.value = "40"
V.mini_batch_string.value = "32"
V.testing_files = ""
V.replicates_string.value = "1"

for batch_seed in ["1", "-1"]:
  V.batch_seed_string.value = batch_seed
  for weights_seed in ["1", "-1"]:
    V.weights_seed_string.value = weights_seed
    V.logs_folder.value = "scratch/seeds/trained-classifier-bs="+batch_seed+"-ws="+weights_seed
    C.train_actuate()

    wait_for_job()

    check_file_exists(os.path.join(V.logs_folder.value, "train1.log"))
    check_file_exists(os.path.join(V.logs_folder.value, "train_1r.log"))
    check_file_exists(os.path.join(V.logs_folder.value,
                                   "train_1r", "vgg.ckpt-"+V.nsteps_string.value+".index"))

run(["hetero", "stop"], stdout=PIPE, stderr=STDOUT)


import tensorflow as tf
import numpy as np

same_weights = ["scratch/seeds/trained-classifier-bs=1-ws=1/train_1r/vgg.ckpt-0",
                "scratch/seeds/trained-classifier-bs=-1-ws=1/train_1r/vgg.ckpt-0"]
diff_weights = ["scratch/seeds/trained-classifier-bs=1-ws=-1/train_1r/vgg.ckpt-0",
                "scratch/seeds/trained-classifier-bs=-1-ws=-1/train_1r/vgg.ckpt-0"]

model0_var_names = None
for model in [*same_weights, *diff_weights]:
  model_var_names = tf.train.list_variables(model)
  if not model0_var_names:
    model0_var_names = model_var_names
    continue
  if model0_var_names != model_var_names:
    print("ERROR: var list not the same")
    print(model+" vars: "+str(model_var_names))

for var_name in model0_var_names:
  if any(x in var_name[0] for x in ["Adam", "global_step", "beta1_power", "beta2_power"]):
    continue
  if len(var_name[1])==1:  # biases
    continue
  same0_var_value = tf.train.load_variable(same_weights[0], var_name[0])
  for model in same_weights[1:]:
    same_var_value = tf.train.load_variable(model, var_name[0])
    if not np.all(same0_var_value == same_var_value):
      print("ERROR: "+model+" "+var_name[0]+" not the same")
  for model in diff_weights:
    diff_var_value = tf.train.load_variable(model, var_name[0])
    if not np.any(same0_var_value != diff_var_value):
      print("ERROR: "+model+" "+var_name[0]+" not different")
