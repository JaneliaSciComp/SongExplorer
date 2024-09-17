#!/usr/bin/env python3

# test that the cluster, snippets, and context windows work

import sys
import os
import shutil
import glob
from subprocess import run, PIPE, STDOUT
import asyncio

import numpy as np
import bokeh.events as be

from libtest import wait_for_job, check_file_exists, count_lines_with_label, count_lines, get_srcrepobindirs

_, repo_path, bindirs = get_srcrepobindirs()

os.environ['PATH'] = os.pathsep.join([*os.environ['PATH'].split(os.pathsep), *bindirs])
  
sys.path.append(os.path.join(repo_path, "src", "gui"))
import model as M
import view as V
import controller as C

os.makedirs(os.path.join(repo_path, "test", "scratch", "visualization"))
shutil.copy(os.path.join(repo_path, "configuration.py"),
            os.path.join(repo_path, "test", "scratch", "visualization"))

M.init(None, os.path.join(repo_path, "test", "scratch", "visualization", "configuration.py"), True)
V.init(None)
C.init(None)

M.deterministic='1'

os.makedirs(os.path.join(repo_path, "test", "scratch", "visualization", "groundtruth-data", "round1"))
shutil.copy(os.path.join(repo_path, "data", "PS_20130625111709_ch3.wav"), \
            os.path.join(repo_path, "test", "scratch", "visualization", "groundtruth-data", "round1"))
shutil.copy(os.path.join(repo_path, "data", "PS_20130625111709_ch3-annotated-person1.csv"), \
            os.path.join(repo_path, "test", "scratch", "visualization", "groundtruth-data", "round1"))

run(["hstart", "1,0,1"])

V.groundtruth_folder.value = os.path.join(repo_path, "test", "scratch", "visualization", "groundtruth-data")
V.kinds_touse.value = 'annotated'
V.labels_touse.value = 'sine,pulse,ambient'
V.groundtruth_update()

if not V.recordings.options == ['', os.path.join("round1", "PS_20130625111709_ch3.wav")]:
    print("ERROR: recordings pull down after changing groundtruth folder")

V.recordings.value = os.path.join("round1", "PS_20130625111709_ch3.wav")
C.recordings_callback(None,None,1)

C.panright_callback()

V.context.value = "204.8"
V.shiftby.value = "0.0"
V.optimizer.value = "Adam"
V.learning_rate.value = "0.0002"
V.model_parameters["dropout"].value = "50"
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
V.model_parameters["window"].value = "6.4"
V.model_parameters["stride"].value = "1.6"
V.model_parameters["mel_dct"].value = "7,7"
V.model_parameters["range"].value = ""
V.logs_folder.value = os.path.join(repo_path, "test", "scratch", "visualization", "untrained-classifier")
V.labels_touse.value = "mel-pulse,mel-sine,ambient"
V.kinds_touse.value = "annotated"
V.nsteps.value = "0"
V.restore_from.value = ""
V.save_and_validate_period.value = "0"
V.validate_percentage.value = "0"
V.mini_batch.value = "32"
V.test_files.value = ""
V.batch_seed.value = "1"
V.weights_seed.value = "1"
V.nreplicates.value = "1"

asyncio.run(C.train_actuate())

wait_for_job(M.status_ticker_queue)

check_file_exists(os.path.join(V.logs_folder.value, "train1.log"))
check_file_exists(os.path.join(V.logs_folder.value, "train_1r.log"))
check_file_exists(os.path.join(V.logs_folder.value, "train_1r",
                               "ckpt-"+V.nsteps.value+".index"))

V.model_file.value = os.path.join(repo_path, "test", "scratch", "visualization", "untrained-classifier", \
                                  "train_"+V.nreplicates.value+"r", \
                                  "ckpt-"+V.nsteps.value+".index")
V.kinds_touse.value = "annotated"
V.activations_equalize_ratio.value = "1000"
V.activations_max_sounds.value = "10000"
asyncio.run(C.activations_actuate())

wait_for_job(M.status_ticker_queue)

check_file_exists(os.path.join(V.groundtruth_folder.value, "activations.log"))
check_file_exists(os.path.join(V.groundtruth_folder.value, "activations.npz"))

V.cluster_these_layers.value = ["2","3"]
V.cluster_parameters["ndims"].value = "2"
V.cluster_parameters["pca-fraction"].value = "1.0"
V.cluster_parameters["neighbors"].value = "10"
V.cluster_parameters["distance"].value = "0.1"
M.pca_batch_size = "0"
M.cluster_parallelize=1
asyncio.run(C.cluster_actuate())

wait_for_job(M.status_ticker_queue)

check_file_exists(os.path.join(V.groundtruth_folder.value, "cluster.log"))
check_file_exists(os.path.join(V.groundtruth_folder.value, "cluster.npz"))

asyncio.run(C.visualize_actuate())

wait_for_job(M.status_ticker_queue)

if len(M.clustered_sounds)!=99:
    print("ERROR: # of clustered sounds after visualization")

ilayer = int(V.cluster_these_layers.value[0])
C.layer_callback(M.layers[ilayer])

npzfile = np.load(os.path.join(V.groundtruth_folder.value, "cluster.npz"), allow_pickle=True)
event = npzfile['activations_clustered'][ilayer][0,:]
C.cluster_tap_callback(event)

event = be.Event()
event.x = M.snippets_pix/2
event.y = 1
C.snippets_tap_callback(event)

C.panright_callback()

run(["hstop"], stdout=PIPE, stderr=STDOUT)
