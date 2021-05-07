#!/bin/bash

# prepare the best network to use as a classifier

# freeze.sh <context-ms> <model-architecture> <model-parameters> <logdir> <model> <check-point> <parallelize> <audio-tic-rate> <audio-nchannels>

# e.g.
# $SONGEXPLORER_BIN freeze.sh 204.8 convolutional '{"representation":"waveform", "window_ms":6.4, "stride_ms":1.6, "mel_dct":"7,7", "dropout":0.5, "kernel_sizes":5,3,3", last_conv_width":130, "nfeatures":"256,256,256", "dilate_after_layer":65535, "stride_after_layer":65535, "connection_type":"plain"}' `pwd`/trained-classifier 1k 50 16384 5000 1

context_ms=$1
architecture=$2
model_parameters=$3
logdir=$4
model=$5
check_point=$6
parallelize=$7
audio_tic_rate=$8
audio_nchannels=$9

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

echo SongExplorer version: $(cat $DIR/../VERSION.txt)

readarray -t labels_touse < $logdir/$model/labels.txt
labels_touse_str=$(IFS=, ; echo "${labels_touse[*]}")

expr="/usr/bin/python3 $DIR/speech_commands_custom/freeze.py \
      --start_checkpoint=$logdir/$model/ckpt-${check_point} \
      --output_file=$logdir/$model/frozen-graph.ckpt-${check_point}.pb \
      --labels_touse=$labels_touse_str \
      --context_ms=$context_ms \
      --parallelize=$parallelize \
      --audio_tic_rate=$audio_tic_rate \
      --nchannels=$audio_nchannels \
      --model_architecture=${architecture} \
      --model_parameters='$model_parameters'"

cmd="date; \
     hostname; \
     $expr &> $logdir/$model/frozen-graph.ckpt-${check_point}.log; \
     sync; \
     date"
echo $cmd

eval "$cmd"
