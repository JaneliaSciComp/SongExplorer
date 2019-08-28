#!/bin/bash

# save hidden layer activations at annotated time points

# hidden.sh <config-file> <context_ms> <shiftby_ms> <window_ms> <mel> <dct> <stride_ms> <kernel_sizes> <last_conv_width> <nfeatures> <logdir> <model> <check-point> <path-to-wavfiles> <label-types> <mini-batch>

# e.g.
# deepsong hidden.sh `pwd`/configuration.sh 204.8 0.0 6.4 7 7 1.6 5,3,3 130 256,256,256 `pwd`/trained-classifier 1k 50 `pwd`/groundtruth-data annotated 32

config_file=$1
context_ms=$2
shiftby_ms=$3
window_ms=$4
mel=$5
dct=$6
stride_ms=$7
kernel_sizes=$8
last_conv_width=$9
nfeatures=${10}
logdir=${11}
model=${12}
check_point=${13}
data_dir=${14}
labels_touse=${15}
mini_batch=${16}

source $config_file

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

readarray -t wanted_words < $logdir/train_$model/vgg_labels.txt
wanted_words_str=$(IFS=, ; echo "${wanted_words[*]}")

expr="IFS=','; \
      /usr/bin/python3 $DIR/speech_commands_custom/train.py \
     --data_url= --data_dir=$data_dir \
     --wanted_words=$wanted_words_str \
     --labels_touse=$labels_touse \
     --how_many_training_steps=$((check_point + 1)) \
     --learning_rate=0 \
     --save_step_interval=0 \
     --eval_step_interval=0 \
     --start_checkpoint=$logdir/train_${model}/vgg.ckpt-$check_point \
     \
     --train_dir=$logdir/train_$model \
     --summaries_dir=$logdir/retrain_logs_$model \
     --sample_rate=$tic_rate \
     --clip_duration_ms=$context_ms \
     --window_size_ms=$window_ms \
     --window_stride_ms=$stride_ms \
     --background_frequency=0.0 \
     --silence_percentage=0.0 \
     --unknown_percentage=0.0 \
     --validation_percentage=0.0 \
     --validation_offset_percentage=0.0 \
     --testing_percentage=100 \
     --time_shift_ms=$shiftby_ms \
     --time_shift_random False \
     --dct_coefficient_count=$dct \
     --filterbank_channel_count=$mel \
     --model_architecture=vgg \
     --filter_counts $nfeatures \
     --filter_sizes $kernel_sizes \
     --final_filter_len $last_conv_width \
     --batch_size=$mini_batch \
     --save_hidden=True"

cmd="date;
     hostname;
     $expr &> $data_dir/hidden-samples.log;
     date"
     #wait;
echo $cmd

logfile=$data_dir/hidden.log
jobname=train-$model

hidden_it "$cmd" "$logfile" "$jobname"
