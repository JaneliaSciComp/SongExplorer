#!/bin/bash

# generate a .tf log file of per-class probabilities

# classify1.sh <context-ms> '' <logdir> <model> <check-point> <wavfile> <audio-tic-rate> <parallelize>

# e.g.
# $SONGEXPLORER_BIN classify1.sh 204.8 '' `pwd`/trained-classifier train_1 50 `pwd`/groundtruth-data/round1/20161207T102314_ch1_p1.wav 5000 65536

context_ms=$1
dummy_placeholder=$2
logdir=$3
model=$4
check_point=$5
wavfile=$6
audio_tic_rate=$7
parallelize=$8

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

echo SongExplorer version: $(cat $DIR/../VERSION.txt)

model_str=$logdir/$model/frozen-graph.ckpt-${check_point}.pb
expr="/usr/bin/python3 $DIR/speech_commands_custom/infer-dense.py \
      --model=$model_str \
      --labels=$logdir/$model/labels.txt \
      --wav=$wavfile \
      --verbose \
      --context_ms=$context_ms \
      --parallelize=$parallelize"

wavdir=`dirname $wavfile`
wavbase=`basename $wavfile`
tffile=$wavdir/${wavbase%.*}.tf

cmd="date; \
     hostname; \
     echo $CUDA_VISIBLE_DEVICES; \
     nvidia-smi; \
     $expr &> $tffile; \
     sync; \
     date"
echo $cmd

eval "$cmd"
