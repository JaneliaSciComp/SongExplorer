#!/usr/bin/env python

# recapitulate the tutorial via the python interface

import sys
import os
import shutil
import glob
from subprocess import run, PIPE, STDOUT
import asyncio
import re

from libtest import wait_for_job, check_file_exists, count_lines_with_label, count_lines, get_srcrepobindirs

_, repo_path, bindirs = get_srcrepobindirs()

os.environ['PATH'] = os.pathsep.join([*bindirs, *os.environ['PATH'].split(os.pathsep)])
  
sys.path.append(os.path.join(repo_path, "src", "gui"))
import model as M
import view as V
import controller as C

os.makedirs(os.path.join(repo_path, "test", "scratch", "tutorial-py"))
shutil.copy(os.path.join(repo_path, "configuration.py"),
            os.path.join(repo_path, "test", "scratch", "tutorial-py"))

M.init(None, os.path.join(repo_path, "test", "scratch", "tutorial-py", "configuration.py"), True)
V.init(None)
C.init(None)

M.deterministic='1'

os.makedirs(os.path.join(repo_path, "test", "scratch", "tutorial-py", "groundtruth-data", "round1"))
shutil.copy(os.path.join(repo_path, "data", "PS_20130625111709_ch3.wav"), \
            os.path.join(repo_path, "test", "scratch", "tutorial-py", "groundtruth-data", "round1"))

run(["hstart", "1,0,1"])

wavpath = os.path.join(repo_path,
                       "test", "scratch", "tutorial-py", "groundtruth-data", "round1", "PS_20130625111709_ch3.wav")
V.wavcsv_files.value = wavpath
V.detect_parameters["time_sigma"].value = "9,4"
V.detect_parameters["time_smooth"].value = "6.4"
V.detect_parameters["frequency_n"].value = "25.6"
V.detect_parameters["frequency_nw"].value = "4"
V.detect_parameters["frequency_p"].value = "0.1,1.0"
V.detect_parameters["frequency_range"].value = "0-"
V.detect_parameters["frequency_smooth"].value = "25.6"
V.detect_parameters["time_sigma_robust"].value = "median"
asyncio.run(C.detect_actuate())

wait_for_job(M.status_ticker_queue)

check_file_exists(wavpath+"-detect.log")
check_file_exists(wavpath+"-detected.csv")
count_lines_with_label(wavpath+"-detected.csv", "time", 536, "ERROR")
count_lines_with_label(wavpath+"-detected.csv", "frequency", 45, "ERROR")
count_lines_with_label(wavpath+"-detected.csv", "neither", 1635, "ERROR")

shutil.copy(os.path.join(repo_path, "data", "PS_20130625111709_ch3.wav-annotated-person1.csv"),
            os.path.join(repo_path, "test", "scratch", "tutorial-py", "groundtruth-data", "round1"))

V.context.value = "204.8"
V.shiftby.value = "0.0"
V.optimizer.value = "Adam"
V.learning_rate.value = "0.0002"
V.model_parameters["dropout"].value = "50"
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
V.logs_folder.value = os.path.join(repo_path, "test", "scratch", "tutorial-py", "trained-classifier1")
V.groundtruth_folder.value = os.path.join(repo_path, "test", "scratch", "tutorial-py", "groundtruth-data")
V.labels_touse.value = "mel-pulse,mel-sine,ambient"
V.kinds_touse.value = "annotated"
V.nsteps.value = "300"
V.restore_from.value = ""
V.save_and_validate_period.value = "30"
V.validate_percentage.value = "40"
V.mini_batch.value = "32"
V.test_files.value = ""
V.batch_seed.value = "1"
V.weights_seed.value = "1"
V.nreplicates.value = "1"
V.loss.value = "exclusive"

asyncio.run(C.train_actuate())

wait_for_job(M.status_ticker_queue)

check_file_exists(os.path.join(V.logs_folder.value, "train1.log"))
check_file_exists(os.path.join(V.logs_folder.value, "train_1r.log"))
check_file_exists(os.path.join(V.logs_folder.value, "train_1r",
                               "ckpt-"+V.nsteps.value+".index"))
check_file_exists(os.path.join(V.logs_folder.value, "train_1r",
                               "logits.validation.ckpt-"+V.nsteps.value+".npz"))

V.precision_recall_ratios.value = "0.5,1.0,2.0"
asyncio.run(C.accuracy_actuate())

wait_for_job(M.status_ticker_queue)

check_file_exists(os.path.join(V.logs_folder.value, "accuracy.log"))
check_file_exists(os.path.join(V.logs_folder.value, "precision-recall.pdf"))
check_file_exists(os.path.join(V.logs_folder.value, "confusion-matrix.pdf"))
check_file_exists(os.path.join(V.logs_folder.value, "train_1r",
                               "precision-recall.ckpt-"+V.nsteps.value+".pdf"))
check_file_exists(os.path.join(V.logs_folder.value, "train_1r",
                               "probability-density.ckpt-"+V.nsteps.value+".pdf"))
check_file_exists(os.path.join(V.logs_folder.value, "train_1r",
                               "thresholds.ckpt-"+V.nsteps.value+".csv"))
check_file_exists(os.path.join(V.logs_folder.value, "train_1r",
                               "confusion-matrix.ckpt-"+V.nsteps.value+".pdf"))
check_file_exists(os.path.join(V.logs_folder.value, "train-validation-loss.pdf"))
check_file_exists(os.path.join(V.logs_folder.value, "P-R-F1-average.pdf"))
check_file_exists(os.path.join(V.logs_folder.value, "P-R-F1-label.pdf"))
check_file_exists(os.path.join(V.logs_folder.value, "P-R-F1-model.pdf"))
check_file_exists(os.path.join(V.logs_folder.value, "PvR.pdf"))

V.model_file.value = os.path.join(V.logs_folder.value, "train_"+V.nreplicates.value+"r",
                                  "ckpt-"+V.nsteps.value+".index")
asyncio.run(C.freeze_actuate())

wait_for_job(M.status_ticker_queue)

check_file_exists(os.path.join(V.logs_folder.value, "train_1r",
                               "freeze.ckpt-"+V.nsteps.value+".log"))
check_file_exists(os.path.join(V.logs_folder.value, "train_1r",
                               "frozen-graph.ckpt-"+V.nsteps.value+".pb",
                               "saved_model.pb"))

os.makedirs(os.path.join(repo_path, "test", "scratch", "tutorial-py", "groundtruth-data", "round2"))
shutil.copy(os.path.join(repo_path, "data", "20161207T102314_ch1.wav"),
            os.path.join(repo_path, "test", "scratch", "tutorial-py", "groundtruth-data", "round2"))

V.wavcsv_files.value = os.path.join(repo_path,
      "test", "scratch", "tutorial-py", "groundtruth-data", "round2", "20161207T102314_ch1.wav")
V.prevalences.value = ""
asyncio.run(C.classify_actuate())

wait_for_job(M.status_ticker_queue)

wavpath = V.wavcsv_files.value
check_file_exists(wavpath+"-classify.log")
for label in V.labels_touse.value.split(','):
  check_file_exists(wavpath+"-"+label+".wav")

asyncio.run(C.ethogram_actuate())

wait_for_job(M.status_ticker_queue)

check_file_exists(wavpath+"-ethogram.log")
for pr in V.precision_recall_ratios.value.split(','):
  check_file_exists(wavpath+"-predicted-"+pr+"pr.csv")
count_lines_with_label(wavpath+"-predicted-1.0pr.csv", "mel-pulse", 510, "WARNING")
count_lines_with_label(wavpath+"-predicted-1.0pr.csv", "mel-sine", 767, "WARNING")
count_lines_with_label(wavpath+"-predicted-1.0pr.csv", "ambient", 124, "WARNING")

asyncio.run(C.detect_actuate())

wait_for_job(M.status_ticker_queue)

check_file_exists(wavpath+"-detect.log")
check_file_exists(wavpath+"-detected.csv")
count_lines_with_label(wavpath+"-detected.csv", "time", 1298, "ERROR")
count_lines_with_label(wavpath+"-detected.csv", "frequency", 179, "ERROR")

V.wavcsv_files.value = wavpath+"-detected.csv,"+ \
                               wavpath+"-predicted-1.0pr.csv"
asyncio.run(C.misses_actuate())

wait_for_job(M.status_ticker_queue)

check_file_exists(wavpath+"-misses.log")
check_file_exists(wavpath+"-missed.csv")
count_lines_with_label(wavpath+"-missed.csv", "other", 1569, "WARNING")

V.model_file.value = os.path.join(repo_path, "test", "scratch", "tutorial-py", "trained-classifier1", \
                                  "train_"+V.nreplicates.value+"r", \
                                  "ckpt-"+V.nsteps.value+".index")
V.kinds_touse.value = "annotated,missed"
V.activations_equalize_ratio.value = "1000"
V.activations_max_sounds.value = "10000"
asyncio.run(C.activations_actuate())

wait_for_job(M.status_ticker_queue)

check_file_exists(os.path.join(V.groundtruth_folder.value, "activations.log"))
check_file_exists(os.path.join(V.groundtruth_folder.value, "activations.npz"))

V.cluster_these_layers.value = ["2","3"]
V.cluster_parameters["ndims"].value = "3"
V.cluster_parameters["pca-fraction"].value = "1.0"
V.cluster_parameters["neighbors"].value = "10"
V.cluster_parameters["distance"].value = "0.1"
M.pca_batch_size = "0"
M.cluster_parallelize=1
asyncio.run(C.cluster_actuate())

wait_for_job(M.status_ticker_queue)

check_file_exists(os.path.join(V.groundtruth_folder.value, "cluster.log"))
check_file_exists(os.path.join(V.groundtruth_folder.value, "cluster.npz"))

shutil.copy(os.path.join(repo_path, "data", "20161207T102314_ch1.wav-annotated-person1.csv"),
            os.path.join(repo_path, "test", "scratch", "tutorial-py", "groundtruth-data", "round2"))

V.logs_folder.value = os.path.join(repo_path, "test", "scratch", "tutorial-py", "omit-one")
V.validation_files.value = "PS_20130625111709_ch3.wav,20161207T102314_ch1.wav"
asyncio.run(C.leaveout_actuate(False))

wait_for_job(M.status_ticker_queue)

for ifile in range(1,1+len(V.validation_files.value.split(','))):
  check_file_exists(os.path.join(V.logs_folder.value, "generalize"+str(ifile)+".log"))
  check_file_exists(os.path.join(V.logs_folder.value, "generalize_"+str(ifile)+"w.log"))
  check_file_exists(os.path.join(V.logs_folder.value, "generalize_"+str(ifile)+"w",
                                 "ckpt-"+V.nsteps.value+".index"))
  check_file_exists(os.path.join(V.logs_folder.value, "generalize_"+str(ifile)+"w",
                                 "logits.validation.ckpt-"+V.nsteps.value+".npz"))

asyncio.run(C.accuracy_actuate())

wait_for_job(M.status_ticker_queue)

check_file_exists(os.path.join(V.logs_folder.value, "accuracy.log"))
check_file_exists(os.path.join(V.logs_folder.value, "precision-recall.pdf"))
check_file_exists(os.path.join(V.logs_folder.value, "confusion-matrix.pdf"))
for ifile in range(1,1+len(V.validation_files.value.split(','))):
  check_file_exists(os.path.join(V.logs_folder.value, "generalize_"+str(ifile)+"w",
                                 "precision-recall.ckpt-"+V.nsteps.value+".pdf"))
  check_file_exists(os.path.join(V.logs_folder.value, "generalize_"+str(ifile)+"w",
                                 "probability-density.ckpt-"+V.nsteps.value+".pdf"))
  check_file_exists(os.path.join(V.logs_folder.value, "generalize_"+str(ifile)+"w",
                                 "thresholds.ckpt-"+V.nsteps.value+".csv"))
  check_file_exists(os.path.join(V.logs_folder.value, "generalize_"+str(ifile)+"w",
                                 "confusion-matrix.ckpt-"+V.nsteps.value+".pdf"))
check_file_exists(os.path.join(V.logs_folder.value, "train-validation-loss.pdf"))
check_file_exists(os.path.join(V.logs_folder.value, "P-R-F1-average.pdf"))
check_file_exists(os.path.join(V.logs_folder.value, "P-R-F1-label.pdf"))
check_file_exists(os.path.join(V.logs_folder.value, "P-R-F1-model.pdf"))
check_file_exists(os.path.join(V.logs_folder.value, "PvR.pdf"))

nfeaturess = ["32,32", "64,64"]

shutil.copy(os.path.join(repo_path, "data", "PS_20130625111709_ch3.wav-annotated-notsong.csv"),
            os.path.join(repo_path, "test", "scratch", "tutorial-py", "groundtruth-data", "round1"))
shutil.copy(os.path.join(repo_path, "data", "20161207T102314_ch1.wav-annotated-notsong.csv"),
            os.path.join(repo_path, "test", "scratch", "tutorial-py", "groundtruth-data", "round2"))

for loss in ["exclusive", "overlapped"]:

    logdirs_prefix= "nfeatures"+loss
    V.loss.value = loss
    if loss=="exclusive":
        V.labels_touse.value = "mel-pulse,mel-sine,ambient"
    else:
        V.labels_touse.value = "mel-pulse,mel-sine"

    for nfeatures in nfeaturess:
        V.logs_folder.value = os.path.join(repo_path,
                                           "test", "scratch", "tutorial-py", logdirs_prefix+nfeatures.split(',')[0])
        V.model_parameters["nfeatures"].value = nfeatures
        V.kfold.value = "2"
        asyncio.run(C.xvalidate_actuate())

    wait_for_job(M.status_ticker_queue)

    for nfeatures in nfeaturess:
        for ifold in range(1, 1+int(V.kfold.value)):
            check_file_exists(os.path.join(V.logs_folder.value, "xvalidate"+str(ifold)+".log"))
            check_file_exists(os.path.join(V.logs_folder.value, "xvalidate_"+str(ifold)+"k.log"))
            check_file_exists(os.path.join(V.logs_folder.value, "xvalidate_"+str(ifold)+"k",
                                           "ckpt-"+V.nsteps.value+".index"))
            check_file_exists(os.path.join(V.logs_folder.value, "xvalidate_"+str(ifold)+"k",
                                           "logits.validation.ckpt-"+V.nsteps.value+".npz"))

    V.precision_recall_ratios.value = "1.0"
    for nfeatures in nfeaturess:
        V.logs_folder.value = os.path.join(repo_path,
                                           "test", "scratch", "tutorial-py", logdirs_prefix+nfeatures.split(',')[0])
        asyncio.run(C.accuracy_actuate())

    wait_for_job(M.status_ticker_queue)

    for nfeatures in nfeaturess:
        check_file_exists(os.path.join(V.logs_folder.value, "accuracy.log"))
        check_file_exists(os.path.join(V.logs_folder.value, "precision-recall.pdf"))
        check_file_exists(os.path.join(V.logs_folder.value, "confusion-matrix.pdf"))
        for ifold in range(1, 1+int(V.kfold.value)):
            check_file_exists(os.path.join(V.logs_folder.value, "xvalidate_"+str(ifold)+"k",
                                           "precision-recall.ckpt-"+V.nsteps.value+".pdf"))
            check_file_exists(os.path.join(V.logs_folder.value, "xvalidate_"+str(ifold)+"k",
                                           "probability-density.ckpt-"+V.nsteps.value+".pdf"))
            check_file_exists(os.path.join(V.logs_folder.value, "xvalidate_"+str(ifold)+"k",
                                           "thresholds.ckpt-"+V.nsteps.value+".csv"))
            check_file_exists(os.path.join(V.logs_folder.value, "xvalidate_"+str(ifold)+"k",
                                           "confusion-matrix.ckpt-"+V.nsteps.value+".pdf"))
        check_file_exists(os.path.join(V.logs_folder.value, "train-validation-loss.pdf"))
        check_file_exists(os.path.join(V.logs_folder.value, "P-R-F1-average.pdf"))
        check_file_exists(os.path.join(V.logs_folder.value, "P-R-F1-label.pdf"))
        check_file_exists(os.path.join(V.logs_folder.value, "P-R-F1-model.pdf"))
        check_file_exists(os.path.join(V.logs_folder.value, "PvR.pdf"))

    V.logs_folder.value = os.path.join(repo_path, "test", "scratch", "tutorial-py", logdirs_prefix)
    asyncio.run(C.compare_actuate())

    wait_for_job(M.status_ticker_queue)

    check_file_exists(V.logs_folder.value+"-compare.log")
    check_file_exists(V.logs_folder.value+"-compare-PR-classes.pdf")
    check_file_exists(V.logs_folder.value+"-compare-confusion-matrices.pdf")
    check_file_exists(V.logs_folder.value+"-compare-overall-params-speed.pdf")

V.loss.value = "exclusive"
V.labels_touse.value = "mel-pulse,mel-sine,ambient"

asyncio.run(C.mistakes_actuate())

wait_for_job(M.status_ticker_queue)

check_file_exists(os.path.join(V.groundtruth_folder.value, "mistakes.log"))
check_file_exists(os.path.join(V.groundtruth_folder.value, "round1",
                               "PS_20130625111709_ch3-mistakes.csv"))

V.logs_folder.value = os.path.join(repo_path, "test", "scratch", "tutorial-py", "trained-classifier2")
V.kinds_touse.value = "annotated"
V.validate_percentage.value = "20"
asyncio.run(C.train_actuate())

wait_for_job(M.status_ticker_queue)

check_file_exists(os.path.join(V.logs_folder.value, "train1.log"))
check_file_exists(os.path.join(V.logs_folder.value, "train_1r.log"))
check_file_exists(os.path.join(V.logs_folder.value, "train_1r",
                               "ckpt-"+V.nsteps.value+".index"))
check_file_exists(os.path.join(V.logs_folder.value, "train_1r",
                               "logits.validation.ckpt-"+V.nsteps.value+".npz"))

asyncio.run(C.accuracy_actuate())

wait_for_job(M.status_ticker_queue)

check_file_exists(os.path.join(V.logs_folder.value, "accuracy.log"))
check_file_exists(os.path.join(V.logs_folder.value, "precision-recall.pdf"))
check_file_exists(os.path.join(V.logs_folder.value, "confusion-matrix.pdf"))
check_file_exists(os.path.join(V.logs_folder.value, "train_1r",
                               "precision-recall.ckpt-"+V.nsteps.value+".pdf"))
check_file_exists(os.path.join(V.logs_folder.value, "train_1r",
                               "probability-density.ckpt-"+V.nsteps.value+".pdf"))
check_file_exists(os.path.join(V.logs_folder.value, "train_1r",
                               "thresholds.ckpt-"+V.nsteps.value+".csv"))
check_file_exists(os.path.join(V.logs_folder.value, "train_1r",
                               "confusion-matrix.ckpt-"+V.nsteps.value+".pdf"))
check_file_exists(os.path.join(V.logs_folder.value, "train-validation-loss.pdf"))
check_file_exists(os.path.join(V.logs_folder.value, "P-R-F1-average.pdf"))
check_file_exists(os.path.join(V.logs_folder.value, "P-R-F1-label.pdf"))
check_file_exists(os.path.join(V.logs_folder.value, "P-R-F1-model.pdf"))
check_file_exists(os.path.join(V.logs_folder.value, "PvR.pdf"))

V.model_file.value = os.path.join(V.logs_folder.value, "train_"+V.nreplicates.value+"r",
                                  "ckpt-"+V.nsteps.value+".index")
asyncio.run(C.freeze_actuate())

wait_for_job(M.status_ticker_queue)

check_file_exists(os.path.join(V.logs_folder.value, "train_1r",
                               "freeze.ckpt-"+V.nsteps.value+".log"))
check_file_exists(os.path.join(V.logs_folder.value, "train_1r",
                               "frozen-graph.ckpt-"+V.nsteps.value+".pb",
                               "saved_model.pb"))

os.mkdir(os.path.join(repo_path, "test", "scratch", "tutorial-py", "groundtruth-data", "dense"))
shutil.copy(os.path.join(repo_path, "data", "20190122T093303a-7.wav"),
            os.path.join(repo_path, "test", "scratch", "tutorial-py", "groundtruth-data", "dense"))

V.wavcsv_files.value = os.path.join(repo_path,
      "test", "scratch", "tutorial-py", "groundtruth-data", "dense", "20190122T093303a-7.wav")
asyncio.run(C.classify_actuate())

wait_for_job(M.status_ticker_queue)

wavpath = V.wavcsv_files.value
check_file_exists(wavpath+"-classify.log")
for label in V.labels_touse.value.split(','):
  check_file_exists(wavpath+"-"+label+".wav")

asyncio.run(C.ethogram_actuate())

wait_for_job(M.status_ticker_queue)

check_file_exists(wavpath+"-ethogram.log")
for pr in V.precision_recall_ratios.value.split(','):
  check_file_exists(wavpath+"-predicted-"+pr+"pr.csv")

shutil.copy(os.path.join(repo_path, "data", "20190122T093303a-7.wav-annotated-person2.csv"),
            os.path.join(repo_path, "test", "scratch", "tutorial-py", "groundtruth-data", "dense"))
shutil.copy(os.path.join(repo_path, "data", "20190122T093303a-7.wav-annotated-person3.csv"),
            os.path.join(repo_path, "test", "scratch", "tutorial-py", "groundtruth-data", "dense"))

V.test_files.value = ""
V.validation_files.value = "20190122T093303a-7.wav"
V.congruence_portion.value = "union"
V.congruence_convolve.value = "0.0"
V.congruence_measure.value = "both"
asyncio.run(C.congruence_actuate())

wait_for_job(M.status_ticker_queue)

congruence_dir0 = next(filter(lambda x: re.fullmatch('congruence-[0-9]{8}T[0-9]{6}', x),
                              os.listdir(V.groundtruth_folder.value)))
congruence_dir = "congruence-11112233T445566"
os.rename(os.path.join(V.groundtruth_folder.value, congruence_dir0),
          os.path.join(V.groundtruth_folder.value, congruence_dir))
wavpath = V.validation_files.value
check_file_exists(os.path.join(V.groundtruth_folder.value, congruence_dir, "congruence.log"))
check_file_exists(os.path.join(V.groundtruth_folder.value, congruence_dir, "dense",
                               wavpath+"-disjoint-everyone.csv"))
kinds = ["tic", "label"]
persons = ["person2", "person3"]
for kind in kinds:
  for label in V.labels_touse.value.split(','):
    check_file_exists(os.path.join(V.groundtruth_folder.value, congruence_dir,
                                   "congruence."+kind+"."+label+".csv"))
    count_lines(os.path.join(V.groundtruth_folder.value, congruence_dir,
                             "congruence."+kind+"."+label+".csv"), M.nprobabilities+2)
    check_file_exists(os.path.join(V.groundtruth_folder.value, congruence_dir,
                                   "congruence."+kind+"."+label+".pdf"))
  for pr in V.precision_recall_ratios.value.split(','):
    for label in V.labels_touse.value.split(','):
      check_file_exists(os.path.join(V.groundtruth_folder.value, congruence_dir,
                                     "congruence."+kind+"."+label+"."+pr+"pr-venn.pdf"))
      check_file_exists(os.path.join(V.groundtruth_folder.value, congruence_dir,
                                     "congruence."+kind+"."+label+"."+pr+"pr.pdf"))
    check_file_exists(os.path.join(V.groundtruth_folder.value, congruence_dir, "dense",
                                   wavpath+"-disjoint-"+kind+"-not"+pr+"pr.csv"))
    check_file_exists(os.path.join(V.groundtruth_folder.value, congruence_dir, "dense",
                                   wavpath+"-disjoint-"+kind+"-only"+pr+"pr.csv"))
  for person in persons:
    check_file_exists(os.path.join(V.groundtruth_folder.value, congruence_dir, "dense",
                                   wavpath+"-disjoint-"+kind+"-not"+person+".csv"))
    check_file_exists(os.path.join(V.groundtruth_folder.value, congruence_dir, "dense",
                                   wavpath+"-disjoint-"+kind+"-only"+person+".csv"))

V.logs_folder.value = os.path.join(repo_path, "test", "scratch", "tutorial-py", "nfeaturesexclusive64")
V.model_file.value = os.path.join(V.logs_folder.value, "xvalidate_1k", "ckpt-"+V.nsteps.value)+','+ \
                     os.path.join(V.logs_folder.value, "xvalidate_2k", "ckpt-"+V.nsteps.value)

asyncio.run(C.ensemble_actuate())

wait_for_job(M.status_ticker_queue)

check_file_exists(os.path.join(V.logs_folder.value, "xvalidate_1k_2k", "ensemble.log"))
check_file_exists(os.path.join(V.logs_folder.value, "xvalidate_1k_2k",
                               "frozen-graph.ckpt-"+V.nsteps.value+"_"+V.nsteps.value+".pb", "saved_model.pb"))

os.mkdir(os.path.join(repo_path, "test", "scratch", "tutorial-py", "groundtruth-data", "dense-ensemble"))
shutil.copy(os.path.join(repo_path, "data", "20190122T132554a-14.wav"),
            os.path.join(repo_path, "test", "scratch", "tutorial-py", "groundtruth-data", "dense-ensemble"))

V.model_file.value = os.path.join(V.logs_folder.value, "xvalidate_1k_2k",
                                  "frozen-graph.ckpt-"+V.nsteps.value+"_"+V.nsteps.value+".pb")
V.wavcsv_files.value = os.path.join(repo_path,
      "test", "scratch", "tutorial-py", "groundtruth-data", "dense-ensemble", "20190122T132554a-14.wav")
asyncio.run(C.classify_actuate())

wait_for_job(M.status_ticker_queue)

wavpath = V.wavcsv_files.value
check_file_exists(wavpath+"-classify.log")
for label in V.labels_touse.value.split(','):
  check_file_exists(wavpath+"-"+label+".wav")

V.model_file.value = os.path.join(V.logs_folder.value, "xvalidate_1k",
                                  "ckpt-"+V.nsteps.value+".index")
asyncio.run(C.ethogram_actuate())

wait_for_job(M.status_ticker_queue)

check_file_exists(wavpath+"-ethogram.log")
for pr in V.precision_recall_ratios.value.split(','):
  check_file_exists(wavpath+"-predicted-"+pr+"pr.csv")
count_lines_with_label(wavpath+"-predicted-1.0pr.csv", "mel-pulse", 56, "WARNING")
count_lines_with_label(wavpath+"-predicted-1.0pr.csv", "mel-sine", 140, "WARNING")
count_lines_with_label(wavpath+"-predicted-1.0pr.csv", "ambient", 70, "WARNING")

shutil.copy(os.path.join(repo_path, "data", "20190122T132554a-14.wav-annotated-person2.csv"),
            os.path.join(repo_path, "test", "scratch", "tutorial-py", "groundtruth-data", "dense-ensemble"))
shutil.copy(os.path.join(repo_path, "data", "20190122T132554a-14.wav-annotated-person3.csv"),
            os.path.join(repo_path, "test", "scratch", "tutorial-py", "groundtruth-data", "dense-ensemble"))

V.validation_files.value = "20190122T132554a-14.wav"
asyncio.run(C.congruence_actuate())

wait_for_job(M.status_ticker_queue)

congruence_dir0 = next(filter(lambda x: re.fullmatch('congruence-[0-9]{8}T[0-9]{6}', x) and
                                        x != "congruence-11112233T445566",
                              os.listdir(V.groundtruth_folder.value)))
congruence_dir = "congruence-99998877T665544"
os.rename(os.path.join(V.groundtruth_folder.value, congruence_dir0),
          os.path.join(V.groundtruth_folder.value, congruence_dir))
wavpath = V.validation_files.value
check_file_exists(os.path.join(V.groundtruth_folder.value, congruence_dir, "congruence.log"))
check_file_exists(os.path.join(V.groundtruth_folder.value, congruence_dir, "dense-ensemble",
                               wavpath+"-disjoint-everyone.csv"))
for kind in kinds:
  for label in V.labels_touse.value.split(','):
    check_file_exists(os.path.join(V.groundtruth_folder.value, congruence_dir,
                                   "congruence."+kind+"."+label+".csv"))
    count_lines(os.path.join(V.groundtruth_folder.value, congruence_dir,
                                   "congruence."+kind+"."+label+".csv"), M.nprobabilities+2)
    check_file_exists(os.path.join(V.groundtruth_folder.value, congruence_dir,
                                   "congruence."+kind+"."+label+".pdf"))
  for pr in V.precision_recall_ratios.value.split(','):
    for label in V.labels_touse.value.split(','):
      check_file_exists(os.path.join(V.groundtruth_folder.value, congruence_dir,
                                     "congruence."+kind+"."+label+"."+pr+"pr-venn.pdf"))
      check_file_exists(os.path.join(V.groundtruth_folder.value, congruence_dir,
                                     "congruence."+kind+"."+label+"."+pr+"pr.pdf"))
    check_file_exists(os.path.join(V.groundtruth_folder.value, congruence_dir, "dense-ensemble",
                                   wavpath+"-disjoint-"+kind+"-not"+pr+"pr.csv"))
    check_file_exists(os.path.join(V.groundtruth_folder.value, congruence_dir, "dense-ensemble",
                                   wavpath+"-disjoint-"+kind+"-only"+pr+"pr.csv"))
  for person in persons:
    check_file_exists(os.path.join(V.groundtruth_folder.value, congruence_dir, "dense-ensemble",
                                   wavpath+"-disjoint-"+kind+"-not"+person+".csv"))
    check_file_exists(os.path.join(V.groundtruth_folder.value, congruence_dir, "dense-ensemble",
                                   wavpath+"-disjoint-"+kind+"-only"+person+".csv"))

V.model_file.value = os.path.join(V.logs_folder.value, "xvalidate_1k_2k",
                                  "frozen-graph.ckpt-"+V.nsteps.value+"_"+V.nsteps.value+".pb")
V.wavcsv_files.value = os.path.join(repo_path,
      "test", "scratch", "tutorial-py", "groundtruth-data", "round1", "PS_20130625111709_ch3.wav")
asyncio.run(C.classify_actuate())

wait_for_job(M.status_ticker_queue)

wavpath = V.wavcsv_files.value
check_file_exists(wavpath+"-classify.log")
for label in V.labels_touse.value.split(','):
  check_file_exists(wavpath+"-"+label+".wav")

frompath=os.path.join(V.logs_folder.value, "xvalidate_1k")
thresholds_dense_file=next(filter(lambda x: x.startswith('thresholds-dense'),
                                  os.listdir(frompath)))
shutil.move(os.path.join(frompath, thresholds_dense_file),
            os.path.join(V.logs_folder.value, "xvalidate_1k_2k"))

V.model_file.value = os.path.join(V.logs_folder.value, "xvalidate_1k_2k", thresholds_dense_file)
asyncio.run(C.ethogram_actuate())

wait_for_job(M.status_ticker_queue)

check_file_exists(wavpath+"-ethogram.log")
for pr in V.precision_recall_ratios.value.split(','):
  check_file_exists(wavpath+"-predicted-"+pr+"pr.csv")
count_lines_with_label(wavpath+"-predicted-1.0pr.csv", "mel-pulse", 594, "WARNING")

run(["hstop"], stdout=PIPE, stderr=STDOUT)
