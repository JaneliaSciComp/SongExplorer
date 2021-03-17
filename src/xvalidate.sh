#!/bin/bash

# train several networks on different subsets of the annotations

# xvalidate.sh <model-architecture> <context-ms> <shiftby-ms> <representation> <window-ms> <mel> <dct> <stride_ms> <dropout> <optimizer> <learning-rate> <kernel-sizes> <last-conv-width> <nfeatures> <dilate-after-layer> <stride-after-layer> <connection-type> <logdir> <path-to-groundtruth> <word1>,<word2>,...,<wordN> <label-types> <nsteps> <restore-from> <save-and-test-interval> <mini-batch> <testing-files> <audio-tic-rate> <audio-nchannels> <batch-seed> <weights-seed> <kfold> <ifolds>

# e.g.
# $SONGEXPLORER_BIN convolutional xvalidate.sh 204.8 0.0 6.4 7 7 1.6 0.5 adam 0.0002 5,3,3 130 256,256,256 65535 65535 plain `pwd`/cross-validate `pwd`/groundtruth-data mel-pulse,mel-sine,ambient,other annotated 50 '' 10 32 "" 5000 1 -1 -1 8 1,2

architecture=$1
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
data_dir=${19}
wanted_words=${20}
labels_touse=${21}
nsteps=${22}
restore_from=${23}
save_and_test_interval=${24}
mini_batch=${25}
testing_files=${26}
audio_tic_rate=${27}
audio_nchannels=${28}
batch_seed=${29}
weights_seed=${30}
kfold=${31}
ifolds=${32}

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

mkdir -p $logdir

if [ -z "$restore_from" ] ; then
  redirect='&>'
  start_checkpoint=
else
  redirect='&>>'
  start_checkpoint=$logdir/xvalidate_MODEL/${architecture}.ckpt-$restore_from
fi

cmd="date; hostname; echo $CUDA_VISIBLE_DEVICES; nvidia-smi; "
    
ifolds=${ifolds},
while [[ $ifolds =~ .*,.* ]] ; do
    ifold=${ifolds%%,*}
    ifolds=${ifolds#*,}
    model=${ifold}k
    kpercent=$(dc -e "3 k 100 $kfold / p")
    koffset=$(dc -e "$kpercent $ifold 1 - * p")
    expr="/usr/bin/python3 $DIR/speech_commands_custom/train.py \
          --data_url= --data_dir=$data_dir \
          --wanted_words=$wanted_words \
          --labels_touse=$labels_touse \
          --how_many_training_steps=$nsteps \
          --start_checkpoint=${start_checkpoint/MODEL/$model} \
          --save_step_interval=$save_and_test_interval \
          --eval_step_interval=$save_and_test_interval \
          --train_dir=$logdir/xvalidate_$model \
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
          --validation_percentage=$kpercent \
          --validation_offset_percentage=$koffset \
          --testing_percentage=0 \
          --testing_files=$testing_files \
          --time_shift_ms=$shiftby_ms \
          --time_shift_random False \
          --filterbank_channel_count=$mel \
          --dct_coefficient_count=$dct \
          --model_architecture=$architecture \
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

          #--subsample_word='mel-pulse' \
          #--subsample_skip=1 \

    cmd=${cmd}" $expr $redirect $logdir/xvalidate_${model}.log & "
done

cmd=${cmd}"wait; sync; date"
echo $cmd

eval "$cmd"
