#!/bin/bash

# train several networks withholding different subsets of the recordings to test upon

# generalize.sh <context-ms> <shiftby-ms> <representation> <window-ms> <mel> <dct> <stride_ms> <dropout> <optimizer> <learning-rate> <kernel-sizes> <last-conv-width> <nfeatures> <dilate-after-layer> <stride-after-layer> <connection-type> <logdir> <path-to-groundtruth> <word1>,<word2>,...,<wordN> <label-types> <nsteps> <restore-from> <save-and-test-interval> <mini-batch> <testing-files> <audio-tic-rate> <audio-nchannels> <batch-seed> <weights-seed> <ioffset> <subset1> [<subset2> [<subset3>]...]

# e.g.
# $SONGEXPLORER_BIN generalize.sh 204.8 0.0 6.4 7 7 1.6 0.5 adam 0.0002 5,3,3 130 256,256,256 65535 65535 plain `pwd`/leave-one-out `pwd`/groundtruth-data mel-pulse,mel-sine,ambient,other annotated 50 '' 10 32 "" 5000 1 -1 -1 3 20161207T102314_ch1_p1.wav,20161207T102314_ch1_p2.wav,20161207T102314_ch1_p3.wav PS_20130625111709_ch3_p1.wav,PS_20130625111709_ch3_p2.wav,PS_20130625111709_ch3_p3.wav

context_ms=$1
shiftby_ms=$2
representation=$3
window_ms=$4
mel=$5
dct=$6
stride_ms=$7
dropout=$8
optimizer=$9
learning_rate=${10}
kernel_sizes=${11}
last_conv_width=${12}
nfeatures=${13}
dilate_after_layer=${14}
stride_after_layer=${15}
connection_type=${16}
logdir=${17}
data_dir=${18}
wanted_words=${19}
labels_touse=${20}
nsteps=${21}
restore_from=${22}
save_and_test_interval=${23}
mini_batch=${24}
testing_files=${25}
audio_tic_rate=${26}
audio_nchannels=${27}
batch_seed=${28}
weights_seed=${29}
ioffset=${30}

shift 30

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

mkdir -p $logdir

if [ -z "$restore_from" ] ; then
  redirect='&>'
  start_checkpoint=
else
  redirect='&>>'
  start_checkpoint=$logdir/generalize_MODEL/vgg.ckpt-$restore_from
fi

cmd="date; hostname; echo $CUDA_VISIBLE_DEVICES; nvidia-smi; "
    
isubset=1
while (( $# > 0 )) ; do
    model=$((ioffset+isubset))w
    expr="/usr/bin/python3 $DIR/speech_commands_custom/train.py \
          --data_url= --data_dir=$data_dir \
          --wanted_words=$wanted_words \
          --labels_touse=$labels_touse \
          --how_many_training_steps=$nsteps \
          --start_checkpoint=${start_checkpoint/MODEL/$model} \
          --save_step_interval=$save_and_test_interval \
          --eval_step_interval=$save_and_test_interval \
          --train_dir=$logdir/generalize_$model \
          --summaries_dir=$logdir/summaries_$model \
          --sample_rate=$audio_tic_rate \
          --nchannels=$audio_nchannels \
          --clip_duration_ms=$context_ms \
          --window_size_ms=$window_ms \
          --window_stride_ms=$stride_ms \
          --learning_rate=$learning_rate \
          --random_seed_batch=$batch_seed \
          --random_seed_weights=$weights_seed \
          --background_frequency=0.0 \
          --silence_percentage=0.0 \
          --unknown_percentage=0.0 \
          --validation_files=$1 \
          --validation_percentage=0 \
          --validation_offset_percentage=0 \
          --testing_percentage=0 \
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

    cmd=${cmd}" $expr $redirect $logdir/generalize_${model}.log & "
    shift 1
    (( isubset++ ))
done

cmd=${cmd}"wait; sync; date"
echo $cmd

eval "$cmd"
