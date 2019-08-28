#!/bin/bash

# prepare the best network to use as a classifier

# freeze.sh <config-file> <context_ms> <window_ms> <mel> <dct> <stride_ms> <kernel_sizes> <last_conv_width> <nfeatures> <logdir> <model> <check-point>

# e.g.
# deepsong freeze.sh `pwd`/configuration.sh 204.8 6.4 7 7 1.6 5,3,3 130 256,256,256 `pwd`/trained-classifier 1k 50

config_file=$1
context_ms=$2
window_ms=$3
mel=$4
dct=$5
stride_ms=$6
kernel_sizes=$7
last_conv_width=$8
nfeatures=$9
logdir=${10}
model=${11}
check_point=${12}

source $config_file
clip_duration=$(dc -e "3 k $context_ms $stride_ms $nstrides 1 - * + p")

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

readarray -t wanted_words < $logdir/train_$model/vgg_labels.txt
wanted_words_str=$(IFS=, ; echo "${wanted_words[*]}")

expr="IFS=','; \
      /usr/bin/python3 $DIR/speech_commands_custom/freeze.py \
      --start_checkpoint=$logdir/train_$model/vgg.ckpt-${check_point} \
      --output_file=$logdir/train_$model/frozen-graph.ckpt-${check_point}.pb \
      --wanted_words=$wanted_words_str \
      --silence_percentage=0.0 \
      --unknown_percentage=0.0 \
      --clip_duration_ms=$clip_duration \
      --window_size_ms=$window_ms \
      --window_stride_ms=$stride_ms \
      --nstrides=$nstrides \
      --sample_rate=$tic_rate \
      --filterbank_channel_count=$mel \
      --dct_coefficient_count=$dct \
      --model_architecture=vgg \
      --filter_counts=$nfeatures \
      --filter_sizes=$kernel_sizes \
      --final_filter_len=$last_conv_width \
      --batch_size=1"

cmd="date;
     hostname;
     $expr &> $logdir/train_$model/frozen-graph.ckpt-${check_point}.log;
     date"
echo $cmd

logfile=$logdir/train_$model/freeze.ckpt-${check_point}.log
jobname=freeze-$logdir

freeze_it "$cmd" "$logfile" "$jobname"
