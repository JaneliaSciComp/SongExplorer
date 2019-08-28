#!/bin/bash

# train several networks withholding different subsets of the recordings to test upon

# generalize.sh <config-file> <context-ms> <shiftby-ms> <representation> <window-ms> <mel> <dct> <stride_ms> <dropout> <optimizer> <learning-rate> <kernel-sizes> <last-conv-width> <nfeatures> <dilate-after-layer> <stride-after-layer> <connection-type> <logdir> <path-to-groundtruth> <word1>,<word2>,...,<wordN> <label-types> <nsteps> <restore-from> <save-and-test-interval> <mini-batch> <testing-files> <subset1> [<subset2> [<subset3>]...]

# e.g.
# $DEEPSONG_BIN generalize.sh `pwd`/configuration.sh 204.8 0.0 6.4 7 7 1.6 0.5 adam 0.0002 5,3,3 130 256,256,256 65535 65535 plain `pwd`/leave-one-out `pwd`/groundtruth-data mel-pulse,mel-sine,ambient,other annotated 50 '' 10 32 "" 20161207T102314_ch1_p1.wav,20161207T102314_ch1_p2.wav,20161207T102314_ch1_p3.wav PS_20130625111709_ch3_p1.wav,PS_20130625111709_ch3_p2.wav,PS_20130625111709_ch3_p3.wav


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
data_dir=${19}
wanted_words=${20}
labels_touse=${21}
nsteps=${22}
restore_from=${23}
save_and_test_interval=${24}
mini_batch=${25}
testing_files=${26}

shift 26

source $config_file

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

if [ "$generalize_gpu" -eq "0" ] ; then
  models_per_job=1
fi

mkdir -p $logdir

if [ -z "$restore_from" ] ; then
  redirect='&>'
  start_checkpoint=
else
  redirect='&>>'
  start_checkpoint=$logdir/generalize_MODEL/vgg.ckpt-$restore_from
fi

for ((iwithhold=0; $#>0; iwithhold+=models_per_job)); do
    cmd="date; hostname; nvidia-smi; "
        
    for ((jwithhold=1; jwithhold<=models_per_job && jwithhold<=$#; jwithhold++)); do
        model=$((iwithhold + jwithhold))w
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
              --random_seed_batch=-1 \
              --random_seed_weights=-1 \
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
    done

    cmd=${cmd}"wait; sync; date"
    echo $cmd

    logfile=$logdir/generalize$((1+iwithhold)).log

    generalize_it "$cmd" "$logfile"

    shift $models_per_job
done
