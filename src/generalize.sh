#!/bin/bash

# train several networks withholding different subsets of the recordings to test upon

# generalize.sh <config-file> <context_ms> <shiftby_ms> <window_ms> <mel> <dct> <stride_ms> <dropout> <optimizer> <learning_rate> <kernel_sizes> <last_conv_width> <nfeatures> <logdir> <path-to-groundtruth> <word1>,<word2>,...,<wordN> <label-types> <nsteps> <save-and-test-interval> <mini-batch> <testing-files> <subset1> [<subset2> [<subset3>]...]

# e.g.
# deepsong generalize.sh `pwd`/configuration.sh 204.8 0.0 6.4 7 7 1.6 0.5 adam 0.0002 5,3,3 130 256,256,256 `pwd`/leave-one-out `pwd`/groundtruth-data mel-pulse,mel-sine,ambient,other annotated 50 10 32 "" 20161207T102314_ch1_p1.wav,20161207T102314_ch1_p2.wav,20161207T102314_ch1_p3.wav PS_20130625111709_ch3_p1.wav,PS_20130625111709_ch3_p2.wav,PS_20130625111709_ch3_p3.wav


config_file=$1
context_ms=$2
shiftby_ms=$3
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
logdir=${14}
data_dir=${15}
wanted_words=${16}
labels_touse=${17}
nsteps=${18}
save_and_test_interval=${19}
mini_batch=${20}
testing_files=${21}

shift 21

source $config_file

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

if [ "$generalize_gpu" -eq "0" ] ; then
  models_per_job=1
fi

mkdir -p $logdir

igpu=0
for ((iwithhold=0; $#>0; iwithhold+=models_per_job)); do
    cmd="date; hostname; nvidia-smi; "
        
    for ((jwithhold=1; jwithhold<=models_per_job && jwithhold<=$#; jwithhold++)); do
        model=$((iwithhold + jwithhold))w
        expr="/usr/bin/python3 $DIR/speech_commands_custom/train.py \
             --data_url= --data_dir=$data_dir \
             --wanted_words=$wanted_words \
             --labels_touse=$labels_touse \
             --how_many_training_steps=$nsteps \
             --save_step_interval=$save_and_test_interval \
             --eval_step_interval=$save_and_test_interval \
             \
             --train_dir=$logdir/train_$model \
             --summaries_dir=$logdir/retrain_logs_$model \
             --sample_rate=$tic_rate \
             --clip_duration_ms=$context_ms \
             --window_size_ms=$window_ms \
             --window_stride_ms=$stride_ms \
             --learning_rate=$learning_rate \
             --background_frequency=0.0 \
             --silence_percentage=0.0 \
             --unknown_percentage=0.0 \
             --validation_files=$1 \
             --validation_percentage=0 \
             --validation_offset_percentage=0 \
             --testing_percentage=0 \
             --testing_files=$testing_files \
             --random_seed=-1 \
             --time_shift_ms=$shiftby_ms \
             --time_shift_random False \
             --filterbank_channel_count=$mel \
             --dct_coefficient_count=$dct \
             --model_architecture=vgg \
             --filter_counts=$nfeatures \
             --filter_sizes=$kernel_sizes \
             --final_filter_len=$last_conv_width \
             --dropout_prob=$dropout \
             --optimizer=$optimizer \
             --batch_size=$mini_batch"

        cmd=${cmd}" $expr &> $logdir/train_${model}.log & "
    done

    cmd=${cmd}"wait; date"
    echo $cmd

    logfile=$logdir/generalize$iwithhold.log
    jobname=generalize-$model

    generalize_it "$cmd" "$logfile" "$jobname" "$igpu"

    ((igpu++))
    shift $models_per_job
done
