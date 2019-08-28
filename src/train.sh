#!/bin/bash

# train a neural network with the annotations

# train.sh <config-file> <context-ms> <shiftby-ms> <representation> <window-ms> <mel> <dct> <stride-ms> <dropout> <optimizer> <learning-rate> <kernel-sizes> <last-conv-width> <nfeatures> <dilate-after-layer> <stride-after-layer> <connection-type> <logdir> <model> <path-to-groundtruth> <word1>,<word2>,...,<wordN> <label-types> <nsteps> <restore-from> <save-and-test-interval> <validation-percentage> <mini-batch> <testing-files>

# e.g.
# $DEEPSONG_BIN train.sh `pwd`/configuration.sh 204.8 0.0 waveform 6.4 7 7 1.6 0.5 adam 0.0002 5,3,3 130 256,256,256 65535 65535 plain `pwd`/trained-classifier 1 `pwd`/groundtruth-data mel-sine,mel-pulse,ambient,other annotated 50 '' 10 40 32 ""

config_file=$1
context_ms=$2
shiftby_ms=$3
representation=$4
window_ms=$5
mel=$6
dct=$7
stride_ms=$8
dropout=$9
optimizer=${10}
learning_rate=${11}
kernel_sizes=${12}
last_conv_width=${13}
nfeatures=${14}
dilate_after_layer=${15}
stride_after_layer=${16}
connection_type=${17}
logdir=${18}
model=${19}
data_dir=${20}
wanted_words=${21}
labels_touse=${22}
nsteps=${23}
restore_from=${24}
save_and_test_interval=${25}
validation_percentage=${26}
mini_batch=${27}
testing_files=${28}

source $config_file

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

mkdir -p $logdir

if [ -z "$restore_from" ] ; then
  redirect='&>'
  start_checkpoint=
else
  redirect='&>>'
  start_checkpoint=$logdir/train_${model}/vgg.ckpt-$restore_from
fi

expr="/usr/bin/python3 $DIR/speech_commands_custom/train.py \
      --data_url= --data_dir=$data_dir \
      --wanted_words=$wanted_words \
      --labels_touse=$labels_touse \
      --how_many_training_steps=$nsteps \
      --start_checkpoint=$start_checkpoint \
      --save_step_interval=$save_and_test_interval \
      --eval_step_interval=$save_and_test_interval \
      --train_dir=$logdir/train_$model \
      --summaries_dir=$logdir/summaries_$model \
      --sample_rate=$audio_tic_rate \
      --nchannels=$audio_nchannels \
      --clip_duration_ms=$context_ms \
      --window_size_ms=$window_ms \
      --window_stride_ms=$stride_ms \
      --learning_rate=${learning_rate} \
      --random_seed_batch=-1 \
      --random_seed_weights=-1 \
      --background_frequency=0.0 \
      --silence_percentage=0.0 \
      --unknown_percentage=0.0 \
      --validation_percentage=$validation_percentage \
      --validation_offset_percentage=0.0 \
      --testing_percentage=0.0 \
      --testing_files=$testing_files \
      --time_shift_ms=$shiftby_ms \
      --time_shift_random False \
      --filterbank_channel_count=$mel \
      --dct_coefficient_count=$dct \
      --model_architecture=vgg \
      --filter_counts=$nfeatures \
      --dilate_after_layer=$dilate_after_layer \
      --stride_after_layer=$stride_after_layer \
      --connection_type=$connection_type \
      --filter_sizes=$kernel_sizes \
      --final_filter_len=$last_conv_width \
      --dropout_prob=$dropout \
      --representation=$representation \
      --optimizer=$optimizer \
      --batch_size=$mini_batch"

      #--save_fingerprints=True \

cmd="date; \
     hostname; \
     nvidia-smi; \
     $expr $redirect $logdir/train_${model}.log; \
     sync; \
     date"
echo $cmd

logfile=$logdir/train${model}.log

train_it "$cmd" "$logfile"
