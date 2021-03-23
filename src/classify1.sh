#!/bin/bash

# generate a .tf log file of per-class probabilities

# classify1.sh <context-ms> '' <representation> <stride-ms> <logdir> <model> <check-point> <wavfile> <audio-tic-rate> <nwindows>

# e.g.
# $SONGEXPLORER_BIN classify1.sh 204.8 '' waveform 1.6 `pwd`/trained-classifier train_1 50 `pwd`/groundtruth-data/round1/20161207T102314_ch1_p1.wav 5000 65536

context_ms=$1
dummy_placeholder=$2
representation=$3
stride_ms=$4
logdir=$5
model=$6
check_point=$7
wavfile=$8
audio_tic_rate=$9
nwindows=${10}

if [ "$representation" == "waveform" ] ; then
  stride_ms=`dc -e "16 k 1000 $audio_tic_rate / p"`
fi
clip_duration_ms=$(dc -e "3 k $context_ms $stride_ms $nwindows 1 - * + p")
clip_stride_ms=$(dc -e "3 k $stride_ms $nwindows 1 - * p")
frozenlog=$logdir/$model/frozen-graph.ckpt-${check_point}.log
ndownsample2=`grep -e 'strides = \[1, 2' -e 'strides = 2' $frozenlog | wc -l`
stride_ms=`dc -e "$stride_ms 2 $ndownsample2 ^ * p"`
clip_stride_ms=`dc -e "$clip_stride_ms $stride_ms + p"`

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

model_str=$logdir/$model/frozen-graph.ckpt-${check_point}.pb
expr="/usr/bin/python3 $DIR/speech_commands_custom/test_streaming_accuracy.py \
      --model=$model_str \
      --labels=$logdir/$model/labels.txt \
      --wav=$wavfile \
      --verbose \
      --clip_duration_ms=$clip_duration_ms \
      --clip_stride_ms=$clip_stride_ms \
      --window_stride_ms=$stride_ms \
      --output_name=output_layer:0"

wavdir=`dirname $wavfile`
wavbase=`basename $wavfile`
tffile=$wavdir/${wavbase%.*}.tf

cmd="date; \
     hostname; \
     echo $CUDA_VISIBLE_DEVICES; \
     nvidia-smi; \
     echo model=$model_str; \
     unset JAVA_HOME; \
     unset TF_CPP_MIN_LOG_LEVEL; \
     ulimit -c 0; \
     $expr &> $tffile; \
     sync; \
     date"
echo $cmd

eval "$cmd"
