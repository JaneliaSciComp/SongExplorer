#!/bin/bash

# prepare the best network to use as a classifier

# freeze.sh <context-ms> <representation> <window-ms> <stride-ms> <mel> <dct> <model-architecture> <model-parameters> <logdir> <model> <check-point> <nwindows> <audio-tic-rate> <audio-nchannels>

# e.g.
# $SONGEXPLORER_BIN freeze.sh 204.8 waveform 6.4 1.6 7 7 convolutional '{"dropout":0.5, "kernel_sizes":5,3,3", last_conv_width":130, "nfeatures":"256,256,256", "dilate_after_layer":65535, "stride_after_layer":65535, "connection_type":"plain"}' `pwd`/trained-classifier 1k 50 65536 5000 1

context_ms=$1
representation=$2
window_ms=$3
stride_ms=$4
mel=$5
dct=$6
architecture=$7
model_parameters=$8
logdir=$9
model=${10}
check_point=${11}
nwindows=${12}
audio_tic_rate=${13}
audio_nchannels=${14}

if [ "$representation" == "waveform" ] ; then
  stride_ms=`dc -e "16 k 1000 $audio_tic_rate / p"`
fi
clip_duration=$(dc -e "3 k $context_ms $stride_ms $nwindows 1 - * + p")

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

readarray -t wanted_words < $logdir/$model/labels.txt
wanted_words_str=$(IFS=, ; echo "${wanted_words[*]}")

expr="/usr/bin/python3 $DIR/speech_commands_custom/freeze.py \
      --start_checkpoint=$logdir/$model/ckpt-${check_point} \
      --output_file=$logdir/$model/frozen-graph.ckpt-${check_point}.pb \
      --wanted_words=$wanted_words_str \
      --clip_duration_ms=$clip_duration \
      --representation=$representation \
      --window_size_ms=$window_ms \
      --window_stride_ms=$stride_ms \
      --nwindows=$nwindows \
      --sample_rate=$audio_tic_rate \
      --nchannels=$audio_nchannels \
      --filterbank_channel_count=$mel \
      --dct_coefficient_count=$dct \
      --model_architecture=${architecture} \
      --model_parameters='$model_parameters' \
      --batch_size=1"

cmd="date; \
     hostname; \
     $expr &> $logdir/$model/frozen-graph.ckpt-${check_point}.log; \
     sync; \
     date"
echo $cmd

eval "$cmd"
