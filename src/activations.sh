#!/bin/bash

# save input, hidden, and output layer activations at specified time points

# activations.sh <context_ms> <shiftby_ms> <model-architecture> <model-parameters> <logdir> <model> <check-point> <path-to-wavfiles> <labels-to-use> <label-types> <equalize-ratio> <max-sounds> <mini-batch> <audio-tic-rate> <audio-nchannels>

# e.g.
# $SONGEXPLORER_BIN activations.sh 204.8 0.0 convolutional '{"representation":"waveform", "window_ms":6.4, "stride_ms":1.6, "mel_dct":"7,7", "dropout":0.5, "kernel_sizes":5,3,3", last_conv_width":130, "nfeatures":"256,256,256", "dilate_after_layer":65535, "stride_after_layer":65535, "connection_type":"plain"}' `pwd`/trained-classifier 1k 50 `pwd`/groundtruth-data mel-sine,mel-pulse,ambient,other annotated 1000 10000 32 5000 1

context_ms=$1
shiftby_ms=$2
architecture=$3
model_parameters=$4
logdir=$5
model=$6
check_point=$7
data_dir=$8
labels_touse=$9
kinds_touse=${10}
equalize_ratio=${11}
max_sounds=${12}
mini_batch=${13}
audio_tic_rate=${14}
audio_nchannels=${15}

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
      --validation_percentage=0.0 \
      --validation_offset_percentage=0.0 \
      --testing_equalize_ratio=$equalize_ratio \
      --testing_max_sounds=$max_sounds \
      --shiftby_ms=$shiftby_ms \
      --model_architecture=$architecture \
      --model_parameters='$model_parameters' \
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
