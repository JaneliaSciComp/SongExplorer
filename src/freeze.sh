#!/bin/bash

# prepare the best network to use as a classifier

# freeze.sh <context-ms> <representation> <window-ms> <stride-ms> <mel> <dct> <kernel-sizes> <last-conv-width> <nfeatures> <dilate-after-layer> <stride-after-layer> <connection-type> <logdir> <model> <check-point> <nwindows> <audio-tic-rate> <audio-nchannels>

# e.g.
# $SONGEXPLORER_BIN freeze.sh 204.8 waveform 6.4 1.6 7 7 5,3,3 130 256,256,256 65535 65535 plain `pwd`/trained-classifier 1k 50 65536 5000 1

context_ms=$1
representation=$2
window_ms=$3
stride_ms=$4
mel=$5
dct=$6
kernel_sizes=$7
last_conv_width=$8
nfeatures=$9
dilate_after_layer=${10}
stride_after_layer=${11}
connection_type=${12}
logdir=${13}
model=${14}
check_point=${15}
nwindows=${16}
audio_tic_rate=${17}
audio_nchannels=${18}

if [ "$representation" == "waveform" ] ; then
  stride_ms=`dc -e "16 k 1000 $audio_tic_rate / p"`
fi
clip_duration=$(dc -e "3 k $context_ms $stride_ms $nwindows 1 - * + p")

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
      --nwindows=$nwindows \
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

eval "$cmd"
