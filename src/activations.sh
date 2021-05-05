#!/bin/bash

# save input, hidden, and output layer activations at specified time points

# activations.sh <context_ms> <shiftby_ms> <representation> <window_ms> <stride_ms> <mel> <dct> <model-architecture> <model-parameters> <logdir> <model> <check-point> <path-to-wavfiles> <labels-to-use> <label-types> <equalize-ratio> <max-sounds> <mini-batch> <audio-tic-rate> <audio-nchannels>

# e.g.
# $SONGEXPLORER_BIN activations.sh 204.8 0.0 6.4 1.6 7 7 convolutional '{"dropout":0.5, "kernel_sizes":5,3,3", last_conv_width":130, "nfeatures":"256,256,256", "dilate_after_layer":65535, "stride_after_layer":65535, "connection_type":"plain"}' `pwd`/trained-classifier 1k 50 `pwd`/groundtruth-data mel-sine,mel-pulse,ambient,other annotated 1000 10000 32 5000 1

context_ms=$1
shiftby_ms=$2
representation=$3
window_ms=$4
stride_ms=$5
mel=$6
dct=$7
architecture=$8
model_parameters=$9
logdir=${10}
model=${11}
check_point=${12}
data_dir=${13}
labels_touse=${14}
kinds_touse=${15}
equalize_ratio=${16}
max_sounds=${17}
mini_batch=${18}
audio_tic_rate=${19}
audio_nchannels=${20}

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

echo SongExplorer version: $(cat $DIR/../VERSION.txt)

expr="/usr/bin/python3 $DIR/speech_commands_custom/infer-sparse.py \
      --data_dir=$data_dir \
      --labels_touse=$labels_touse \
      --kinds_touse=$kinds_touse \
      --start_checkpoint=$logdir/${model}/ckpt-$check_point \
      --audio_tic_rate=$audio_tic_rate \
      --nchannels=$audio_nchannels \
      --context_ms=$context_ms \
      --window_ms=$window_ms \
      --stride_ms=$stride_ms \
      --validation_percentage=0.0 \
      --validation_offset_percentage=0.0 \
      --testing_equalize_ratio=$equalize_ratio \
      --testing_max_sounds=$max_sounds \
      --shiftby_ms=$shiftby_ms \
      --dct_ncoefficients=$dct \
      --filterbank_nchannels=$mel \
      --model_architecture=$architecture \
      --model_parameters='$model_parameters' \
      --representation=$representation \
      --batch_size=$mini_batch \
      --save_activations=True"

cmd="date; \
     hostname; \
     echo $CUDA_VISIBLE_DEVICES; \
     nvidia-smi; \
     $expr &> $data_dir/activations-sounds.log; \
     sync; \
     date"
echo $cmd

eval "$cmd"
