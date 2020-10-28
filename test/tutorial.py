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

from lib import wait_for_job, check_file_exists, count_lines_with_word, count_lines

repo_path = os.path.dirname(sys.path[0])
  
sys.path.append(os.path.join(repo_path, "src/gui"))
import model as M
import view as V
import controller as C

os.makedirs(os.path.join(repo_path, "test/scratch/py"))
shutil.copy(os.path.join(repo_path, "configuration.pysh"),
            os.path.join(repo_path, "test/scratch/py"))

M.init(None, os.path.join(repo_path, "test/scratch/py/configuration.pysh"))
V.init(None)
C.init(None)

os.makedirs(os.path.join(repo_path, "test/scratch/py/groundtruth-data/round1"))
shutil.copy(os.path.join(repo_path, "data/PS_20130625111709_ch3.wav"), \
            os.path.join(repo_path, "test/scratch/py/groundtruth-data/round1"))

run(["hetero", "start", "1", "1", "1"])

wavpath_noext = os.path.join(repo_path,
                             "test/scratch/py/groundtruth-data/round1/PS_20130625111709_ch3")
V.wavtfcsvfiles_string.value = wavpath_noext+".wav"
V.time_sigma_string.value = "6,3"
V.time_smooth_ms_string.value = "6.4"
V.frequency_n_ms_string.value = "25.6"
V.frequency_nw_string.value = "4"
V.frequency_p_string.value = "0.1,1.0"
V.frequency_smooth_ms_string.value = "25.6"
asyncio.run(C.detect_actuate())

wait_for_job(M.status_ticker_queue)

check_file_exists(wavpath_noext+"-detect.log")
check_file_exists(wavpath_noext+"-detected.csv")
count_lines_with_word(wavpath_noext+"-detected.csv", "time", 543)
count_lines_with_word(wavpath_noext+"-detected.csv", "frequency", 45)
count_lines_with_word(wavpath_noext+"-detected.csv", "ambient", 1138)

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
V.logs_folder.value = os.path.join(repo_path, "test/scratch/py/untrained-classifier")
V.groundtruth_folder.value = os.path.join(repo_path, "test/scratch/py/groundtruth-data")
V.wantedwords_string.value = "time,frequency"
V.labeltypes_string.value = "detected"
V.nsteps_string.value = "0"
V.restore_from_string.value = ""
V.save_and_validate_period_string.value = "0"
V.validate_percentage_string.value = "0"
V.mini_batch_string.value = "32"
V.testing_files = ""
V.batch_seed_string.value = "1"
V.weights_seed_string.value = "1"
V.replicates_string.value = "1"
asyncio.run(C.train_actuate())

wait_for_job(M.status_ticker_queue)

check_file_exists(os.path.join(V.logs_folder.value, "train1.log"))
check_file_exists(os.path.join(V.logs_folder.value, "train_1r.log"))
check_file_exists(os.path.join(V.logs_folder.value,
                               "train_1r","vgg.ckpt-"+V.nsteps_string.value+".index"))

V.model_file.value = os.path.join(repo_path, "test/scratch/py/untrained-classifier",
                                  "train_"+V.replicates_string.value+"r",
                                  "vgg.ckpt-"+V.nsteps_string.value+".meta")
V.activations_equalize_ratio_string.value = "1000"
V.activations_max_samples_string.value = "10000"
asyncio.run(C.activations_actuate())

wait_for_job(M.status_ticker_queue)

check_file_exists(os.path.join(V.groundtruth_folder.value, "activations.log"))
check_file_exists(os.path.join(V.groundtruth_folder.value, "activations-samples.log"))
check_file_exists(os.path.join(V.groundtruth_folder.value, "activations.npz"))

V.cluster_these_layers.value = ["0"]
V.pca_fraction_variance_to_retain_string.value = "0.99"
M.pca_batch_size = "0"
V.cluster_algorithm.value = "tSNE 2D"
V.tsne_perplexity_string.value = "30"
V.tsne_exaggeration_string.value = "12"
asyncio.run(C.cluster_actuate())

wait_for_job(M.status_ticker_queue)

check_file_exists(os.path.join(V.groundtruth_folder.value, "cluster.log"))
check_file_exists(os.path.join(V.groundtruth_folder.value, "cluster.npz"))
check_file_exists(os.path.join(V.groundtruth_folder.value, "cluster-pca.pdf"))

shutil.copy(os.path.join(repo_path, "data/PS_20130625111709_ch3-annotated-person1.csv"),
            os.path.join(repo_path, "test/scratch/py/groundtruth-data/round1"))

V.logs_folder.value = os.path.join(repo_path, "test/scratch/py/trained-classifier1")
V.wantedwords_string.value = "mel-pulse,mel-sine,ambient"
V.labeltypes_string.value = "annotated"
V.nsteps_string.value = "100"
V.save_and_validate_period_string.value = "10"
V.validate_percentage_string.value = "40"
asyncio.run(C.train_actuate())

wait_for_job(M.status_ticker_queue)

check_file_exists(os.path.join(V.logs_folder.value, "train1.log"))
check_file_exists(os.path.join(V.logs_folder.value, "train_1r.log"))
check_file_exists(os.path.join(V.logs_folder.value, "train_1r",
                               "vgg.ckpt-"+V.nsteps_string.value+".index"))
check_file_exists(os.path.join(V.logs_folder.value, "train_1r",
                               "logits.validation.ckpt-"+V.nsteps_string.value+".npz"))

V.precision_recall_ratios_string.value = "0.5,1.0,2.0"
asyncio.run(C.accuracy_actuate())

wait_for_job(M.status_ticker_queue)

check_file_exists(os.path.join(V.logs_folder.value, "accuracy.log"))
check_file_exists(os.path.join(V.logs_folder.value, "accuracy.pdf"))
check_file_exists(os.path.join(V.logs_folder.value, "train_1r",
                               "precision-recall.ckpt-"+V.nsteps_string.value+".pdf"))
check_file_exists(os.path.join(V.logs_folder.value, "train_1r",
                               "probability-density.ckpt-"+V.nsteps_string.value+".pdf"))
check_file_exists(os.path.join(V.logs_folder.value, "train_1r",
                               "thresholds.ckpt-"+V.nsteps_string.value+".csv"))
check_file_exists(os.path.join(V.logs_folder.value, "train-loss.pdf"))
check_file_exists(os.path.join(V.logs_folder.value, "validation-F1.pdf"))
for word in V.wantedwords_string.value.split(','):
  check_file_exists(os.path.join(V.logs_folder.value, "validation-PvR-"+word+".pdf"))

V.model_file.value = os.path.join(V.logs_folder.value, "train_"+V.replicates_string.value+"r",
                                  "vgg.ckpt-"+V.nsteps_string.value+".meta")
asyncio.run(C.freeze_actuate())

wait_for_job(M.status_ticker_queue)

check_file_exists(os.path.join(V.logs_folder.value, "train_1r",
                               "freeze.ckpt-"+V.nsteps_string.value+".log"))
check_file_exists(os.path.join(V.logs_folder.value, "train_1r",
                               "frozen-graph.ckpt-"+V.nsteps_string.value+".pb"))

os.makedirs(os.path.join(repo_path, "test/scratch/py/groundtruth-data/round2"))
shutil.copy(os.path.join(repo_path, "data/20161207T102314_ch1.wav"),
            os.path.join(repo_path, "test/scratch/py/groundtruth-data/round2"))

V.wavtfcsvfiles_string.value = os.path.join(repo_path,
      "test/scratch/py/groundtruth-data/round2/20161207T102314_ch1.wav")
V.prevalences_string.value = ""
asyncio.run(C.classify_actuate())

wait_for_job(M.status_ticker_queue)

wavpath_noext = V.wavtfcsvfiles_string.value[:-4]
check_file_exists(wavpath_noext+".tf")
check_file_exists(wavpath_noext+"-classify1.log")
check_file_exists(wavpath_noext+"-classify2.log")
for word in V.wantedwords_string.value.split(','):
  check_file_exists(wavpath_noext+"-"+word+".wav")

asyncio.run(C.ethogram_actuate())

wait_for_job(M.status_ticker_queue)

check_file_exists(wavpath_noext+"-ethogram.log")
for pr in V.precision_recall_ratios_string.value.split(','):
  check_file_exists(wavpath_noext+"-predicted-"+pr+"pr.csv")
count_lines_with_word(wavpath_noext+"-predicted-1.0pr.csv", "mel-pulse", 1010)
count_lines_with_word(wavpath_noext+"-predicted-1.0pr.csv", "mel-sine", 958)
count_lines_with_word(wavpath_noext+"-predicted-1.0pr.csv", "ambient", 88)

asyncio.run(C.detect_actuate())

wait_for_job(M.status_ticker_queue)

check_file_exists(wavpath_noext+"-detect.log")
check_file_exists(wavpath_noext+"-detected.csv")
count_lines_with_word(wavpath_noext+"-detected.csv", "time", 1309)
count_lines_with_word(wavpath_noext+"-detected.csv", "frequency", 179)

V.wavtfcsvfiles_string.value = wavpath_noext+"-detected.csv,"+ \
                               wavpath_noext+"-predicted-1.0pr.csv"
asyncio.run(C.misses_actuate())

wait_for_job(M.status_ticker_queue)

check_file_exists(wavpath_noext+"-misses.log")
check_file_exists(wavpath_noext+"-missed.csv")
count_lines_with_word(wavpath_noext+"-missed.csv", "other", 2199)

os.mkdir(os.path.join(V.groundtruth_folder.value, "round1", "cluster"))
for file in glob.glob(os.path.join(V.groundtruth_folder.value, "activations*")):
    shutil.move(file, os.path.join(V.groundtruth_folder.value, "round1", "cluster"))

for file in glob.glob(os.path.join(V.groundtruth_folder.value, "cluster*")):
    shutil.move(file, os.path.join(V.groundtruth_folder.value, "round1", "cluster"))

V.model_file.value = os.path.join(repo_path, "test/scratch/py/trained-classifier1", \
                                  "train_"+V.replicates_string.value+"r", \
                                  "vgg.ckpt-"+V.nsteps_string.value+".meta")
V.labeltypes_string.value = "annotated,missed"
V.activations_equalize_ratio_string.value = "1000"
V.activations_max_samples_string.value = "10000"
asyncio.run(C.activations_actuate())

wait_for_job(M.status_ticker_queue)

check_file_exists(os.path.join(V.groundtruth_folder.value, "activations.log"))
check_file_exists(os.path.join(V.groundtruth_folder.value, "activations-samples.log"))
check_file_exists(os.path.join(V.groundtruth_folder.value, "activations.npz"))

V.cluster_these_layers.value = ["2","3"]
V.pca_fraction_variance_to_retain_string.value = "1.0"
M.pca_batch_size = "0"
V.cluster_algorithm.value = "UMAP 3D"
M.cluster_parallelize=1
V.umap_neighbors_string.value = "10"
V.umap_distance_string.value = "0.1"
asyncio.run(C.cluster_actuate())

wait_for_job(M.status_ticker_queue)

check_file_exists(os.path.join(V.groundtruth_folder.value, "cluster.log"))
check_file_exists(os.path.join(V.groundtruth_folder.value, "cluster.npz"))

shutil.copy(os.path.join(repo_path, "data/20161207T102314_ch1-annotated-person1.csv"),
            os.path.join(repo_path, "test/scratch/py/groundtruth-data/round2"))

V.logs_folder.value = os.path.join(repo_path, "test/scratch/py/omit-one")
V.validationfiles_string.value = "PS_20130625111709_ch3.wav,20161207T102314_ch1.wav"
asyncio.run(C.leaveout_actuate(False))

wait_for_job(M.status_ticker_queue)

for ifile in range(1,1+len(V.validationfiles_string.value.split(','))):
  check_file_exists(os.path.join(V.logs_folder.value, "generalize"+str(ifile)+".log"))
  check_file_exists(os.path.join(V.logs_folder.value, "generalize_"+str(ifile)+"w.log"))
  check_file_exists(os.path.join(V.logs_folder.value, "generalize_"+str(ifile)+"w",
                                 "vgg.ckpt-"+V.nsteps_string.value+".index"))
  check_file_exists(os.path.join(V.logs_folder.value, "generalize_"+str(ifile)+"w",
                                 "logits.validation.ckpt-"+V.nsteps_string.value+".npz"))

asyncio.run(C.accuracy_actuate())

wait_for_job(M.status_ticker_queue)

check_file_exists(os.path.join(V.logs_folder.value, "accuracy.log"))
check_file_exists(os.path.join(V.logs_folder.value, "accuracy.pdf"))
check_file_exists(os.path.join(V.logs_folder.value, "confusion-matrices.pdf"))
for ifile in range(1,1+len(V.validationfiles_string.value.split(','))):
  check_file_exists(os.path.join(V.logs_folder.value, "generalize_"+str(ifile)+"w",
                                 "precision-recall.ckpt-"+V.nsteps_string.value+".pdf"))
  check_file_exists(os.path.join(V.logs_folder.value, "generalize_"+str(ifile)+"w",
                                 "probability-density.ckpt-"+V.nsteps_string.value+".pdf"))
  check_file_exists(os.path.join(V.logs_folder.value, "generalize_"+str(ifile)+"w",
                                 "thresholds.ckpt-"+V.nsteps_string.value+".csv"))
check_file_exists(os.path.join(V.logs_folder.value, "train-loss.pdf"))
check_file_exists(os.path.join(V.logs_folder.value, "validation-F1.pdf"))
for word in V.wantedwords_string.value.split(','):
  check_file_exists(os.path.join(V.logs_folder.value, "validation-PvR-"+word+".pdf"))

nfeaturess = ["32,32,32", "64,64,64"]

for nfeatures in nfeaturess:
  V.logs_folder.value = os.path.join(repo_path,
                                     "test/scratch/py/nfeatures-"+nfeatures.split(',')[0])
  V.nfeatures_string.value = nfeatures
  V.kfold_string.value = "2"
  asyncio.run(C.xvalidate_actuate())

wait_for_job(M.status_ticker_queue)

for nfeatures in nfeaturess:
  for ifold in range(1, 1+int(V.kfold_string.value)):
    check_file_exists(os.path.join(V.logs_folder.value, "xvalidate"+str(ifold)+".log"))
    check_file_exists(os.path.join(V.logs_folder.value, "xvalidate_"+str(ifold)+"k.log"))
    check_file_exists(os.path.join(V.logs_folder.value, "xvalidate_"+str(ifold)+"k",
                                   "vgg.ckpt-"+V.nsteps_string.value+".index"))
    check_file_exists(os.path.join(V.logs_folder.value, "xvalidate_"+str(ifold)+"k",
                                   "logits.validation.ckpt-"+V.nsteps_string.value+".npz"))

for nfeatures in nfeaturess:
  V.logs_folder.value = os.path.join(repo_path,
                                     "test/scratch/py/nfeatures-"+nfeatures.split(',')[0])
  asyncio.run(C.accuracy_actuate())

wait_for_job(M.status_ticker_queue)

for nfeatures in nfeaturess:
  check_file_exists(os.path.join(V.logs_folder.value, "accuracy.log"))
  check_file_exists(os.path.join(V.logs_folder.value, "accuracy.pdf"))
  check_file_exists(os.path.join(V.logs_folder.value, "confusion-matrices.pdf"))
  for ifold in range(1, 1+int(V.kfold_string.value)):
    check_file_exists(os.path.join(V.logs_folder.value, "xvalidate_"+str(ifold)+"k",
                                   "precision-recall.ckpt-"+V.nsteps_string.value+".pdf"))
    check_file_exists(os.path.join(V.logs_folder.value, "xvalidate_"+str(ifold)+"k",
                                   "probability-density.ckpt-"+V.nsteps_string.value+".pdf"))
    check_file_exists(os.path.join(V.logs_folder.value, "xvalidate_"+str(ifold)+"k",
                                   "thresholds.ckpt-"+V.nsteps_string.value+".csv"))
  check_file_exists(os.path.join(V.logs_folder.value, "train-loss.pdf"))
  check_file_exists(os.path.join(V.logs_folder.value, "validation-F1.pdf"))
  for word in V.wantedwords_string.value.split(','):
    check_file_exists(os.path.join(V.logs_folder.value, "validation-PvR-"+word+".pdf"))

V.logs_folder.value = os.path.join(repo_path, "test/scratch/py/nfeatures")
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

V.logs_folder.value = os.path.join(repo_path, "test/scratch/py/trained-classifier2")
V.labeltypes_string.value = "annotated"
V.nsteps_string.value = "100"
V.validate_percentage_string.value = "20"
asyncio.run(C.train_actuate())

wait_for_job(M.status_ticker_queue)

check_file_exists(os.path.join(V.logs_folder.value, "train1.log"))
check_file_exists(os.path.join(V.logs_folder.value, "train_1r.log"))
check_file_exists(os.path.join(V.logs_folder.value, "train_1r",
                               "vgg.ckpt-"+V.nsteps_string.value+".index"))
check_file_exists(os.path.join(V.logs_folder.value, "train_1r",
                               "logits.validation.ckpt-"+V.nsteps_string.value+".npz"))

V.precision_recall_ratios_string.value = "1.0"
asyncio.run(C.accuracy_actuate())

wait_for_job(M.status_ticker_queue)

check_file_exists(os.path.join(V.logs_folder.value, "accuracy.log"))
check_file_exists(os.path.join(V.logs_folder.value, "accuracy.pdf"))
check_file_exists(os.path.join(V.logs_folder.value, "train_1r",
                               "precision-recall.ckpt-"+V.nsteps_string.value+".pdf"))
check_file_exists(os.path.join(V.logs_folder.value, "train_1r",
                               "probability-density.ckpt-"+V.nsteps_string.value+".pdf"))
check_file_exists(os.path.join(V.logs_folder.value, "train_1r",
                               "thresholds.ckpt-"+V.nsteps_string.value+".csv"))
check_file_exists(os.path.join(V.logs_folder.value, "train-loss.pdf"))
check_file_exists(os.path.join(V.logs_folder.value, "validation-F1.pdf"))
for word in V.wantedwords_string.value.split(','):
  check_file_exists(os.path.join(V.logs_folder.value, "validation-PvR-"+word+".pdf"))

V.model_file.value = os.path.join(V.logs_folder.value, "train_"+V.replicates_string.value+"r",
                                  "vgg.ckpt-"+V.nsteps_string.value+".meta")
asyncio.run(C.freeze_actuate())

wait_for_job(M.status_ticker_queue)

check_file_exists(os.path.join(V.logs_folder.value, "train_1r",
                               "freeze.ckpt-"+V.nsteps_string.value+".log"))
check_file_exists(os.path.join(V.logs_folder.value, "train_1r",
                               "frozen-graph.ckpt-"+V.nsteps_string.value+".pb"))

os.mkdir(os.path.join(repo_path, "test/scratch/py/groundtruth-data/congruence"))
shutil.copy(os.path.join(repo_path, "data/20190122T093303a-7.wav"),
            os.path.join(repo_path, "test/scratch/py/groundtruth-data/congruence"))

V.wavtfcsvfiles_string.value = os.path.join(repo_path,
      "test/scratch/py/groundtruth-data/congruence/20190122T093303a-7.wav")
asyncio.run(C.classify_actuate())

wait_for_job(M.status_ticker_queue)

wavpath_noext = V.wavtfcsvfiles_string.value[:-4]
check_file_exists(wavpath_noext+"-classify1.log")
check_file_exists(wavpath_noext+".tf")
check_file_exists(wavpath_noext+"-classify2.log")
for word in V.wantedwords_string.value.split(','):
  check_file_exists(wavpath_noext+"-"+word+".wav")

asyncio.run(C.ethogram_actuate())

wait_for_job(M.status_ticker_queue)

check_file_exists(wavpath_noext+"-ethogram.log")
for pr in V.precision_recall_ratios_string.value.split(','):
  check_file_exists(wavpath_noext+"-predicted-"+pr+"pr.csv")

shutil.copy(os.path.join(repo_path, "data/20190122T093303a-7-annotated-person2.csv"),
            os.path.join(repo_path, "test/scratch/py/groundtruth-data/congruence"))
shutil.copy(os.path.join(repo_path, "data/20190122T093303a-7-annotated-person3.csv"),
            os.path.join(repo_path, "test/scratch/py/groundtruth-data/congruence"))

V.testfiles_string.value = ""
V.validationfiles_string.value = "20190122T093303a-7.wav"
asyncio.run(C.congruence_actuate())

wait_for_job(M.status_ticker_queue)

wavpath_noext = V.validationfiles_string.value[:-4]
check_file_exists(os.path.join(V.groundtruth_folder.value, "congruence.log"))
check_file_exists(os.path.join(V.groundtruth_folder.value, "congruence",
                               wavpath_noext+"-disjoint-everyone.csv"))
kinds = ["tic", "word"]
persons = ["person2", "person3"]
for kind in kinds:
  for word in V.wantedwords_string.value.split(','):
    check_file_exists(os.path.join(V.groundtruth_folder.value,
                                   "congruence."+kind+"."+word+".csv"))
    count_lines(os.path.join(V.groundtruth_folder.value,
                                   "congruence."+kind+"."+word+".csv"), M.nprobabilities+2)
    check_file_exists(os.path.join(V.groundtruth_folder.value,
                                   "congruence."+kind+"."+word+".pdf"))
  for pr in V.precision_recall_ratios_string.value.split(','):
    for word in V.wantedwords_string.value.split(','):
      check_file_exists(os.path.join(V.groundtruth_folder.value, "congruence.bar-venn",
                                     "congruence."+kind+"."+word+"."+pr+"pr-venn.pdf"))
      check_file_exists(os.path.join(V.groundtruth_folder.value, "congruence.bar-venn",
                                     "congruence."+kind+"."+word+"."+pr+"pr.pdf"))
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
