#!/usr/bin/python3

# recapitulate the tutorial via the python interface

# export SINGULARITYENV_SONGEXPLORER_STATE=/tmp
# ${SONGEXPLORER_BIN/-B/-B /tmp:/opt/songexplorer/test/scratch -B} test/tutorial.py

import sys
import os
import shutil
import glob
from subprocess import run, PIPE, STDOUT
import asyncio

from lib import wait_for_job, check_file_exists, count_lines_with_label, count_lines

repo_path = os.path.dirname(sys.path[0])
  
sys.path.append(os.path.join(repo_path, "src/gui"))
import model as M
import view as V
import controller as C

os.makedirs(os.path.join(repo_path, "test/scratch/tutorial-py"))
shutil.copy(os.path.join(repo_path, "configuration.pysh"),
            os.path.join(repo_path, "test/scratch/tutorial-py"))

M.init(None, os.path.join(repo_path, "test/scratch/tutorial-py/configuration.pysh"))
V.init(None)
C.init(None)

M.deterministic='1'

os.makedirs(os.path.join(repo_path, "test/scratch/tutorial-py/groundtruth-data/round1"))
shutil.copy(os.path.join(repo_path, "data/PS_20130625111709_ch3.wav"), \
            os.path.join(repo_path, "test/scratch/tutorial-py/groundtruth-data/round1"))

run(["hetero", "start", "1", "1", "1"])

wavpath_noext = os.path.join(repo_path,
                             "test/scratch/tutorial-py/groundtruth-data/round1/PS_20130625111709_ch3")
V.wavcsv_files.value = wavpath_noext+".wav"
V.time_sigma.value = "9,4"
V.time_smooth_ms.value = "6.4"
V.frequency_n_ms.value = "25.6"
V.frequency_nw.value = "4"
V.frequency_p.value = "0.1,1.0"
V.frequency_smooth_ms.value = "25.6"
asyncio.run(C.detect_actuate())

wait_for_job(M.status_ticker_queue)

check_file_exists(wavpath_noext+"-detect.log")
check_file_exists(wavpath_noext+"-detected.csv")
count_lines_with_label(wavpath_noext+"-detected.csv", "time", 536, "ERROR")
count_lines_with_label(wavpath_noext+"-detected.csv", "frequency", 45, "ERROR")
count_lines_with_label(wavpath_noext+"-detected.csv", "neither", 1635, "ERROR")

V.context_ms.value = "204.8"
V.shiftby_ms.value = "0.0"
V.optimizer.value = "Adam"
V.learning_rate.value = "0.0002"
V.model_parameters["dropout"].value = "0.5"
V.model_parameters["kernel_sizes"].value = "5,3"
V.model_parameters["nlayers"].value = "2"
V.model_parameters["nfeatures"].value = "64,64"
V.model_parameters["dilate_after_layer"].value = "65535"
V.model_parameters["stride_after_layer"].value = "65535"
V.model_parameters["connection_type"].value = "plain"
V.model_parameters["representation"].value = "mel-cepstrum"
V.model_parameters["window_ms"].value = "6.4"
V.model_parameters["stride_ms"].value = "1.6"
V.model_parameters["mel_dct"].value = "7,7"
V.logs_folder.value = os.path.join(repo_path, "test/scratch/tutorial-py/untrained-classifier")
V.groundtruth_folder.value = os.path.join(repo_path, "test/scratch/tutorial-py/groundtruth-data")
V.labels_touse.value = "time,frequency"
V.kinds_touse.value = "detected"
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
check_file_exists(os.path.join(V.logs_folder.value,
                               "train_1r","ckpt-"+V.nsteps.value+".index"))

V.model_file.value = os.path.join(repo_path, "test/scratch/tutorial-py/untrained-classifier",
                                  "train_"+V.nreplicates.value+"r",
                                  "ckpt-"+V.nsteps.value+".meta")
V.activations_equalize_ratio.value = "1000"
V.activations_max_sounds.value = "10000"
asyncio.run(C.activations_actuate())

wait_for_job(M.status_ticker_queue)

check_file_exists(os.path.join(V.groundtruth_folder.value, "activations.log"))
check_file_exists(os.path.join(V.groundtruth_folder.value, "activations.npz"))

V.cluster_these_layers.value = ["0"]
V.pca_fraction_variance_to_retain.value = "0.99"
M.pca_batch_size = "0"
V.cluster_algorithm.value = "tSNE 2D"
V.tsne_perplexity.value = "30"
V.tsne_exaggeration.value = "12"
asyncio.run(C.cluster_actuate())

wait_for_job(M.status_ticker_queue)

check_file_exists(os.path.join(V.groundtruth_folder.value, "cluster.log"))
check_file_exists(os.path.join(V.groundtruth_folder.value, "cluster.npz"))
check_file_exists(os.path.join(V.groundtruth_folder.value, "cluster-pca.pdf"))

shutil.copy(os.path.join(repo_path, "data/PS_20130625111709_ch3-annotated-person1.csv"),
            os.path.join(repo_path, "test/scratch/tutorial-py/groundtruth-data/round1"))

V.logs_folder.value = os.path.join(repo_path, "test/scratch/tutorial-py/trained-classifier1")
V.labels_touse.value = "mel-pulse,mel-sine,ambient"
V.kinds_touse.value = "annotated"
V.nsteps.value = "300"
V.save_and_validate_period.value = "30"
V.validate_percentage.value = "40"
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
check_file_exists(os.path.join(V.logs_folder.value, "accuracy.pdf"))
check_file_exists(os.path.join(V.logs_folder.value, "train_1r",
                               "precision-recall.ckpt-"+V.nsteps.value+".pdf"))
check_file_exists(os.path.join(V.logs_folder.value, "train_1r",
                               "probability-density.ckpt-"+V.nsteps.value+".pdf"))
check_file_exists(os.path.join(V.logs_folder.value, "train_1r",
                               "thresholds.ckpt-"+V.nsteps.value+".csv"))
check_file_exists(os.path.join(V.logs_folder.value, "train-loss.pdf"))
check_file_exists(os.path.join(V.logs_folder.value, "validation-F1.pdf"))
for label in V.labels_touse.value.split(','):
  check_file_exists(os.path.join(V.logs_folder.value, "validation-PvR-"+label+".pdf"))

V.model_file.value = os.path.join(V.logs_folder.value, "train_"+V.nreplicates.value+"r",
                                  "ckpt-"+V.nsteps.value+".meta")
asyncio.run(C.freeze_actuate())

wait_for_job(M.status_ticker_queue)

check_file_exists(os.path.join(V.logs_folder.value, "train_1r",
                               "freeze.ckpt-"+V.nsteps.value+".log"))
check_file_exists(os.path.join(V.logs_folder.value, "train_1r",
                               "frozen-graph.ckpt-"+V.nsteps.value+".pb",
                               "saved_model.pb"))

os.makedirs(os.path.join(repo_path, "test/scratch/tutorial-py/groundtruth-data/round2"))
shutil.copy(os.path.join(repo_path, "data/20161207T102314_ch1.wav"),
            os.path.join(repo_path, "test/scratch/tutorial-py/groundtruth-data/round2"))

V.wavcsv_files.value = os.path.join(repo_path,
      "test/scratch/tutorial-py/groundtruth-data/round2/20161207T102314_ch1.wav")
V.prevalences.value = ""
asyncio.run(C.classify_actuate())

wait_for_job(M.status_ticker_queue)

wavpath_noext = V.wavcsv_files.value[:-4]
check_file_exists(wavpath_noext+"-classify.log")
for label in V.labels_touse.value.split(','):
  check_file_exists(wavpath_noext+"-"+label+".wav")

asyncio.run(C.ethogram_actuate())

wait_for_job(M.status_ticker_queue)

check_file_exists(wavpath_noext+"-ethogram.log")
for pr in V.precision_recall_ratios.value.split(','):
  check_file_exists(wavpath_noext+"-predicted-"+pr+"pr.csv")
count_lines_with_label(wavpath_noext+"-predicted-1.0pr.csv", "mel-pulse", 536, "WARNING")
count_lines_with_label(wavpath_noext+"-predicted-1.0pr.csv", "mel-sine", 518, "WARNING")
count_lines_with_label(wavpath_noext+"-predicted-1.0pr.csv", "ambient", 261, "WARNING")

asyncio.run(C.detect_actuate())

wait_for_job(M.status_ticker_queue)

check_file_exists(wavpath_noext+"-detect.log")
check_file_exists(wavpath_noext+"-detected.csv")
count_lines_with_label(wavpath_noext+"-detected.csv", "time", 1298, "ERROR")
count_lines_with_label(wavpath_noext+"-detected.csv", "frequency", 179, "ERROR")

V.wavcsv_files.value = wavpath_noext+"-detected.csv,"+ \
                               wavpath_noext+"-predicted-1.0pr.csv"
asyncio.run(C.misses_actuate())

wait_for_job(M.status_ticker_queue)

check_file_exists(wavpath_noext+"-misses.log")
check_file_exists(wavpath_noext+"-missed.csv")
count_lines_with_label(wavpath_noext+"-missed.csv", "other", 1460, "WARNING")

os.mkdir(os.path.join(V.groundtruth_folder.value, "round1", "cluster"))
for file in glob.glob(os.path.join(V.groundtruth_folder.value, "activations*")):
    shutil.move(file, os.path.join(V.groundtruth_folder.value, "round1", "cluster"))

for file in glob.glob(os.path.join(V.groundtruth_folder.value, "cluster*")):
    shutil.move(file, os.path.join(V.groundtruth_folder.value, "round1", "cluster"))

V.model_file.value = os.path.join(repo_path, "test/scratch/tutorial-py/trained-classifier1", \
                                  "train_"+V.nreplicates.value+"r", \
                                  "ckpt-"+V.nsteps.value+".meta")
V.kinds_touse.value = "annotated,missed"
V.activations_equalize_ratio.value = "1000"
V.activations_max_sounds.value = "10000"
asyncio.run(C.activations_actuate())

wait_for_job(M.status_ticker_queue)

check_file_exists(os.path.join(V.groundtruth_folder.value, "activations.log"))
check_file_exists(os.path.join(V.groundtruth_folder.value, "activations.npz"))

V.cluster_these_layers.value = ["2","3"]
V.pca_fraction_variance_to_retain.value = "1.0"
M.pca_batch_size = "0"
V.cluster_algorithm.value = "UMAP 3D"
M.cluster_parallelize=1
V.umap_neighbors.value = "10"
V.umap_distance.value = "0.1"
asyncio.run(C.cluster_actuate())

wait_for_job(M.status_ticker_queue)

check_file_exists(os.path.join(V.groundtruth_folder.value, "cluster.log"))
check_file_exists(os.path.join(V.groundtruth_folder.value, "cluster.npz"))

shutil.copy(os.path.join(repo_path, "data/20161207T102314_ch1-annotated-person1.csv"),
            os.path.join(repo_path, "test/scratch/tutorial-py/groundtruth-data/round2"))

V.logs_folder.value = os.path.join(repo_path, "test/scratch/tutorial-py/omit-one")
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
check_file_exists(os.path.join(V.logs_folder.value, "accuracy.pdf"))
check_file_exists(os.path.join(V.logs_folder.value, "confusion-matrices.pdf"))
for ifile in range(1,1+len(V.validation_files.value.split(','))):
  check_file_exists(os.path.join(V.logs_folder.value, "generalize_"+str(ifile)+"w",
                                 "precision-recall.ckpt-"+V.nsteps.value+".pdf"))
  check_file_exists(os.path.join(V.logs_folder.value, "generalize_"+str(ifile)+"w",
                                 "probability-density.ckpt-"+V.nsteps.value+".pdf"))
  check_file_exists(os.path.join(V.logs_folder.value, "generalize_"+str(ifile)+"w",
                                 "thresholds.ckpt-"+V.nsteps.value+".csv"))
check_file_exists(os.path.join(V.logs_folder.value, "train-loss.pdf"))
check_file_exists(os.path.join(V.logs_folder.value, "validation-F1.pdf"))
for label in V.labels_touse.value.split(','):
  check_file_exists(os.path.join(V.logs_folder.value, "validation-PvR-"+label+".pdf"))

nfeaturess = ["32,32", "64,64"]

for nfeatures in nfeaturess:
  V.logs_folder.value = os.path.join(repo_path,
                                     "test/scratch/tutorial-py/nfeatures-"+nfeatures.split(',')[0])
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

for nfeatures in nfeaturess:
  V.logs_folder.value = os.path.join(repo_path,
                                     "test/scratch/tutorial-py/nfeatures-"+nfeatures.split(',')[0])
  asyncio.run(C.accuracy_actuate())

wait_for_job(M.status_ticker_queue)

for nfeatures in nfeaturess:
  check_file_exists(os.path.join(V.logs_folder.value, "accuracy.log"))
  check_file_exists(os.path.join(V.logs_folder.value, "accuracy.pdf"))
  check_file_exists(os.path.join(V.logs_folder.value, "confusion-matrices.pdf"))
  for ifold in range(1, 1+int(V.kfold.value)):
    check_file_exists(os.path.join(V.logs_folder.value, "xvalidate_"+str(ifold)+"k",
                                   "precision-recall.ckpt-"+V.nsteps.value+".pdf"))
    check_file_exists(os.path.join(V.logs_folder.value, "xvalidate_"+str(ifold)+"k",
                                   "probability-density.ckpt-"+V.nsteps.value+".pdf"))
    check_file_exists(os.path.join(V.logs_folder.value, "xvalidate_"+str(ifold)+"k",
                                   "thresholds.ckpt-"+V.nsteps.value+".csv"))
  check_file_exists(os.path.join(V.logs_folder.value, "train-loss.pdf"))
  check_file_exists(os.path.join(V.logs_folder.value, "validation-F1.pdf"))
  for label in V.labels_touse.value.split(','):
    check_file_exists(os.path.join(V.logs_folder.value, "validation-PvR-"+label+".pdf"))

V.logs_folder.value = os.path.join(repo_path, "test/scratch/tutorial-py/nfeatures")
asyncio.run(C.compare_actuate())

wait_for_job(M.status_ticker_queue)

check_file_exists(V.logs_folder.value+"-compare.log")
check_file_exists(V.logs_folder.value+"-compare-precision-recall.pdf")
check_file_exists(V.logs_folder.value+"-compare-confusion-matrices.pdf")
check_file_exists(V.logs_folder.value+"-compare-overall-params-speed.pdf")

asyncio.run(C.mistakes_actuate())

wait_for_job(M.status_ticker_queue)

check_file_exists(os.path.join(V.groundtruth_folder.value, "mistakes.log"))
check_file_exists(os.path.join(V.groundtruth_folder.value, "round1",
                               "PS_20130625111709_ch3-mistakes.csv"))

V.logs_folder.value = os.path.join(repo_path, "test/scratch/tutorial-py/trained-classifier2")
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

V.precision_recall_ratios.value = "1.0"
asyncio.run(C.accuracy_actuate())

wait_for_job(M.status_ticker_queue)

check_file_exists(os.path.join(V.logs_folder.value, "accuracy.log"))
check_file_exists(os.path.join(V.logs_folder.value, "accuracy.pdf"))
check_file_exists(os.path.join(V.logs_folder.value, "train_1r",
                               "precision-recall.ckpt-"+V.nsteps.value+".pdf"))
check_file_exists(os.path.join(V.logs_folder.value, "train_1r",
                               "probability-density.ckpt-"+V.nsteps.value+".pdf"))
check_file_exists(os.path.join(V.logs_folder.value, "train_1r",
                               "thresholds.ckpt-"+V.nsteps.value+".csv"))
check_file_exists(os.path.join(V.logs_folder.value, "train-loss.pdf"))
check_file_exists(os.path.join(V.logs_folder.value, "validation-F1.pdf"))
for label in V.labels_touse.value.split(','):
  check_file_exists(os.path.join(V.logs_folder.value, "validation-PvR-"+label+".pdf"))

V.model_file.value = os.path.join(V.logs_folder.value, "train_"+V.nreplicates.value+"r",
                                  "ckpt-"+V.nsteps.value+".meta")
asyncio.run(C.freeze_actuate())

wait_for_job(M.status_ticker_queue)

check_file_exists(os.path.join(V.logs_folder.value, "train_1r",
                               "freeze.ckpt-"+V.nsteps.value+".log"))
check_file_exists(os.path.join(V.logs_folder.value, "train_1r",
                               "frozen-graph.ckpt-"+V.nsteps.value+".pb",
                               "saved_model.pb"))

os.mkdir(os.path.join(repo_path, "test/scratch/tutorial-py/groundtruth-data/congruence"))
shutil.copy(os.path.join(repo_path, "data/20190122T093303a-7.wav"),
            os.path.join(repo_path, "test/scratch/tutorial-py/groundtruth-data/congruence"))

V.wavcsv_files.value = os.path.join(repo_path,
      "test/scratch/tutorial-py/groundtruth-data/congruence/20190122T093303a-7.wav")
asyncio.run(C.classify_actuate())

wait_for_job(M.status_ticker_queue)

wavpath_noext = V.wavcsv_files.value[:-4]
check_file_exists(wavpath_noext+"-classify.log")
for label in V.labels_touse.value.split(','):
  check_file_exists(wavpath_noext+"-"+label+".wav")

asyncio.run(C.ethogram_actuate())

wait_for_job(M.status_ticker_queue)

check_file_exists(wavpath_noext+"-ethogram.log")
for pr in V.precision_recall_ratios.value.split(','):
  check_file_exists(wavpath_noext+"-predicted-"+pr+"pr.csv")

shutil.copy(os.path.join(repo_path, "data/20190122T093303a-7-annotated-person2.csv"),
            os.path.join(repo_path, "test/scratch/tutorial-py/groundtruth-data/congruence"))
shutil.copy(os.path.join(repo_path, "data/20190122T093303a-7-annotated-person3.csv"),
            os.path.join(repo_path, "test/scratch/tutorial-py/groundtruth-data/congruence"))

V.test_files.value = ""
V.validation_files.value = "20190122T093303a-7.wav"
V.congruence_portion.value = "union"
V.congruence_convolve.value = "0.0"
asyncio.run(C.congruence_actuate())

wait_for_job(M.status_ticker_queue)

wavpath_noext = V.validation_files.value[:-4]
check_file_exists(os.path.join(V.groundtruth_folder.value, "congruence.log"))
check_file_exists(os.path.join(V.groundtruth_folder.value, "congruence",
                               wavpath_noext+"-disjoint-everyone.csv"))
kinds = ["tic", "label"]
persons = ["person2", "person3"]
for kind in kinds:
  for label in V.labels_touse.value.split(','):
    check_file_exists(os.path.join(V.groundtruth_folder.value,
                                   "congruence."+kind+"."+label+".csv"))
    count_lines(os.path.join(V.groundtruth_folder.value,
                                   "congruence."+kind+"."+label+".csv"), M.nprobabilities+2)
    check_file_exists(os.path.join(V.groundtruth_folder.value,
                                   "congruence."+kind+"."+label+".pdf"))
  for pr in V.precision_recall_ratios.value.split(','):
    for label in V.labels_touse.value.split(','):
      check_file_exists(os.path.join(V.groundtruth_folder.value,
                                     "congruence."+kind+"."+label+"."+pr+"pr-venn.pdf"))
      check_file_exists(os.path.join(V.groundtruth_folder.value,
                                     "congruence."+kind+"."+label+"."+pr+"pr.pdf"))
    check_file_exists(os.path.join(V.groundtruth_folder.value, "congruence",
                                   wavpath_noext+"-disjoint-"+kind+"-not"+pr+"pr.csv"))
    check_file_exists(os.path.join(V.groundtruth_folder.value, "congruence",
                                   wavpath_noext+"-disjoint-"+kind+"-only"+pr+"pr.csv"))
  for person in persons:
    check_file_exists(os.path.join(V.groundtruth_folder.value, "congruence",
                                   wavpath_noext+"-disjoint-"+kind+"-not"+person+".csv"))
    check_file_exists(os.path.join(V.groundtruth_folder.value, "congruence",
                                   wavpath_noext+"-disjoint-"+kind+"-only"+person+".csv"))

run(["hetero", "stop"], stdout=PIPE, stderr=STDOUT)
