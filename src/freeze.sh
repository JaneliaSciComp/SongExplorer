#!/bin/bash

# prepare the best network to use as a classifier

# freeze.sh <config-file> <context_ms> <representation> <window_ms> <stride_ms> <mel> <dct> <kernel_sizes> <last_conv_width> <nfeatures> <dilate-after-layer> <stride-after-layer> <connection-type> <logdir> <model> <check-point>

# e.g.
# $DEEPSONG_BIN freeze.sh `pwd`/configuration.sh 204.8 waveform 6.4 1.6 7 7 5,3,3 130 256,256,256 65535 65535 plain `pwd`/trained-classifier 1k 50

config_file=$1
context_ms=$2
representation=$3
window_ms=$4
stride_ms=$5
mel=$6
dct=$7
kernel_sizes=$8
last_conv_width=$9
nfeatures=${10}
dilate_after_layer=${11}
stride_after_layer=${12}
connection_type=${13}
logdir=${14}
model=${15}
check_point=${16}

source $config_file
if [ "$representation" == "waveform" ] ; then
  stride_ms=`dc -e "16 k 1000 $audio_tic_rate / p"`
fi
clip_duration=$(dc -e "3 k $context_ms $stride_ms $nstrides 1 - * + p")

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

readarray -t wanted_words < $logdir/$model/vgg_labels.txt
wanted_words_str=$(IFS=, ; echo "${wanted_words[*]}")

expr="/usr/bin/python3 $DIR/speech_commands_custom/freeze.py \
      --start_checkpoint=$logdir/$model/vgg.ckpt-${check_point} \
      --output_file=$logdir/$model/frozen-graph.ckpt-${check_point}.pb \
      --wanted_words=$wanted_words_str \
      --silence_percentage=0.0 \
      --unknown_percentage=0.0 \
      --clip_duration_ms=$clip_duration \
      --representation=$representation \
      --window_size_ms=$window_ms \
      --window_stride_ms=$stride_ms \
      --nstrides=$nstrides \
      --sample_rate=$audio_tic_rate \
      --nchannels=$audio_nchannels \
      --filterbank_channel_count=$mel \
      --dct_coefficient_count=$dct \
      --model_architecture=vgg \
      --filter_counts=$nfeatures \
      --dilate_after_layer=$dilate_after_layer \
      --stride_after_layer=$stride_after_layer \
      --connection_type=$connection_type \
      --filter_sizes=$kernel_sizes \
      --final_filter_len=$last_conv_width \
      --batch_size=1"

cmd="date; \
     hostname; \
     $expr &> $logdir/$model/frozen-graph.ckpt-${check_point}.log; \
     sync; \
     date"
echo $cmd

logfile=$logdir/$model/freeze.ckpt-${check_point}.log

freeze_it "$cmd" "$logfile"
