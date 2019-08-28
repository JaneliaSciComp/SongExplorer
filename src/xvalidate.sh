#!/bin/bash

# train several networks on different subsets of the annotations

# xvalidate.sh <config-file> <context_ms> <shiftby_ms> <window_ms> <mel> <dct> <stride_ms> <dropout> <optimizer> <learning_rate> <kernel_sizes> <last_conv_width> <nfeatures> <logdir> <path-to-groundtruth> <word1>,<word2>,...,<wordN> <label-types> <nsteps> <save-and-test-interval> <mini-batch> <kfold> <testing-files>

# e.g.
# deepsong xvalidate.sh `pwd`/configuration.sh 204.8 0.0 6.4 7 7 1.6 0.5 adam 0.0002 5,3,3 130 256,256,256 `pwd`/cross-validate `pwd`/groundtruth-data mel-pulse,mel-sine,ambient,other annotated 50 10 32 8 ""

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
kfold=${21}
testing_files=${22}

source $config_file

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

if [ "$xvalidate_gpu" -eq "0" ] ; then
  models_per_job=1
fi

mkdir -p $logdir

igpu=0
for ((ifold=1; ifold<=kfold;  ifold+=models_per_job)); do
    cmd="date; hostname; nvidia-smi; "
        
    for ((jfold=$ifold; jfold<ifold+models_per_job && jfold<=kfold; jfold++)); do
        model=${jfold}k
        kpercent=$(dc -e "3 k 100 $kfold / p")
        koffset=$(dc -e "$kpercent $jfold 1 - * p")
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
             --validation_percentage=$kpercent \
             --validation_offset_percentage=$koffset \
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

             #--subsample_word='mel-pulse' \
             #--subsample_skip=1 \

             #--start_checkpoint=$logdir/train_${model}/vgg.ckpt-50000 \
             # change &> to &>> for train?k.log
             # change -oo to -o

        cmd=${cmd}" $expr &> $logdir/train_${model}.log & "
    done

    cmd=${cmd}"wait; date"
    echo $cmd

    logfile=$logdir/xvalidate$ifold.log
    jobname=xvalidate-$model

    xvalidate_it "$cmd" "$logfile" "$jobname" "$igpu"
    ((igpu++))
done
