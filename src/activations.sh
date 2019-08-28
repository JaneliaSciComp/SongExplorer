#!/bin/bash

# save input, hidden, and output layer activations at specified time points

# activations.sh <config-file> <context_ms> <shiftby_ms> <representation> <window_ms> <mel> <dct> <stride_ms> <kernel_sizes> <last_conv_width> <nfeatures> <dilate-after-layer> <stride-after-layer> <connection-type> <logdir> <model> <check-point> <path-to-wavfiles> <wanted-words> <label-types> <equalize-ratio> <max-samples> <mini-batch>

# e.g.
# $DEEPSONG_BIN activations.sh `pwd`/configuration.sh 204.8 0.0 6.4 7 7 1.6 5,3,3 130 256,256,256 65535 65535 plain `pwd`/trained-classifier 1k 50 `pwd`/groundtruth-data mel-sine,mel-pulse,ambient,other annotated 1000 10000 32

config_file=$1
context_ms=$2
shiftby_ms=$3
representation=$4
window_ms=$5
mel=$6
dct=$7
stride_ms=$8
kernel_sizes=$9
last_conv_width=${10}
nfeatures=${11}
dilate_after_layer=${12}
stride_after_layer=${13}
connection_type=${14}
logdir=${15}
model=${16}
check_point=${17}
data_dir=${18}
wanted_words=${19}
labels_touse=${20}
equalize_ratio=${21}
max_samples=${22}
mini_batch=${23}

source $config_file

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

expr="/usr/bin/python3 $DIR/speech_commands_custom/infer.py \
      --data_url= --data_dir=$data_dir \
      --wanted_words=$wanted_words \
      --labels_touse=$labels_touse \
      --start_checkpoint=$logdir/${model}/vgg.ckpt-$check_point \
      --sample_rate=$audio_tic_rate \
      --nchannels=$audio_nchannels \
      --clip_duration_ms=$context_ms \
      --window_size_ms=$window_ms \
      --window_stride_ms=$stride_ms \
      --random_seed_batch=-1 \
      --random_seed_weights=-1 \
      --silence_percentage=0.0 \
      --unknown_percentage=0.0 \
      --validation_percentage=0.0 \
      --validation_offset_percentage=0.0 \
      --testing_percentage=100 \
      --testing_equalize_ratio=$equalize_ratio \
      --testing_max_samples=$max_samples \
      --time_shift_ms=$shiftby_ms \
      --time_shift_random False \
      --dct_coefficient_count=$dct \
      --filterbank_channel_count=$mel \
      --model_architecture=vgg \
      --filter_counts $nfeatures \
      --dilate_after_layer=$dilate_after_layer \
      --stride_after_layer=$stride_after_layer \
      --connection_type=$connection_type \
      --filter_sizes $kernel_sizes \
      --final_filter_len $last_conv_width \
      --representation=$representation \
      --batch_size=$mini_batch \
      --save_activations=True"

cmd="date; \
     hostname; \
     $expr &> $data_dir/activations-samples.log; \
     sync; \
     date"
echo $cmd

logfile=$data_dir/activations.log

activations_it "$cmd" "$logfile"
