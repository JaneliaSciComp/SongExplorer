#!/bin/bash

# generate .wav files of per-label probabilities

# classify.sh <context-ms> <shiftby_ms> <logdir> <model> <check-point> <wavfile> <audio-tic-rate> <parallelize> [<labels> <prevalences>]

# e.g.
# $SONGEXPLORER_BIN classify.sh 204.8 0.0 `pwd`/trained-classifier train_1 50 `pwd`/groundtruth-data/round1/20161207T102314_ch1_p1.wav 5000 65536 mel-pulse,mel-sine,ambient 0.1,0.1,0.8

context_ms=$1
shiftby_ms=$2
logdir=$3
model=$4
check_point=$5
wavfile=$6
audio_tic_rate=$7
parallelize=$8
if [[ $# == 10 ]] ; then
  labels=$9
  prevalences=${10}
else
  labels=
  prevalences=
fi

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

echo SongExplorer version: $(cat $DIR/../VERSION.txt)

model_str=$logdir/$model/frozen-graph.ckpt-${check_point}.pb
expr="/usr/bin/python3 $DIR/speech_commands_custom/infer-dense.py \
      --model=$model_str \
      --model_labels=$logdir/$model/labels.txt \
      --labels=$labels \
      --prevalences=$prevalences \
      --wav=$wavfile \
      --verbose \
      --context_ms=$context_ms \
      --shiftby_ms=$shiftby_ms \
      --parallelize=$parallelize"

cmd="date; \
     hostname; \
     echo $CUDA_VISIBLE_DEVICES; \
     nvidia-smi; \
     $expr; \
     sync; \
     date"
echo $cmd

eval "$cmd"
