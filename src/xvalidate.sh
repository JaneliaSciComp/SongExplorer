#!/bin/bash

# train several networks on different subsets of the annotations

# xvalidate.sh <context-ms> <shiftby-ms> <representation> <window-ms> <stride_ms> <mel> <dct> <optimizer> <learning-rate> <model-architecture> <model-parameters-json> <logdir> <path-to-groundtruth> <label1>,<label2>,...,<labelN> <kinds-to-use> <nsteps> <restore-from> <save-and-validate-period> <mini-batch> <testing-files> <audio-tic-rate> <audio-nchannels> <batch-seed> <weights-seed> <kfold> <ifolds>

# e.g.
# $SONGEXPLORER_BIN xvalidate.sh 204.8 0.0 6.4 1.6 7 7 Adam 0.0002 convolutional '{"dropout":0.5, "kernel_sizes":5,3,3", last_conv_width":130, "nfeatures":"256,256,256", "dilate_after_layer":65535, "stride_after_layer":65535, "connection_type":"plain"}' `pwd`/cross-validate `pwd`/groundtruth-data mel-pulse,mel-sine,ambient,other annotated 50 '' 10 32 "" 5000 1 -1 -1 8 1,2

context_ms=$1
shiftby_ms=$2
representation=$3
window_ms=$4
stride_ms=$5
mel=$6
dct=$7
optimizer=$8
learning_rate=$9
architecture=${10}
model_parameters=${11}
logdir=${12}
data_dir=${13}
labels_touse=${14}
kinds_touse=${15}
nsteps=${16}
restore_from=${17}
save_and_validate_period=${18}
mini_batch=${19}
testing_files=${20}
audio_tic_rate=${21}
audio_nchannels=${22}
batch_seed=${23}
weights_seed=${24}
kfold=${25}
ifolds=${26}

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

mkdir -p $logdir

if [ -z "$restore_from" ] ; then
  redirect='&>'
  start_checkpoint=
else
  redirect='&>>'
  start_checkpoint=$logdir/xvalidate_MODEL/ckpt-$restore_from
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
          --data_dir=$data_dir \
          --labels_touse=$labels_touse \
          --kinds_touse=$kinds_touse \
          --how_many_training_steps=$nsteps \
          --start_checkpoint=${start_checkpoint/MODEL/$model} \
          --save_step_period=$save_and_validate_period \
          --validate_step_period=$save_and_validate_period \
          --train_dir=$logdir/xvalidate_$model \
          --summaries_dir=$logdir/summaries_$model \
          --audio_tic_rate=$audio_tic_rate \
          --nchannels=$audio_nchannels \
          --context_ms=$context_ms \
          --window_ms=$window_ms \
          --stride_ms=$stride_ms \
          --learning_rate=$learning_rate \
          --random_seed_batch=$batch_seed \
          --random_seed_weights=$weights_seed \
          --validation_percentage=$kpercent \
          --validation_offset_percentage=$koffset \
          --testing_files=$testing_files \
          --shiftby_ms=$shiftby_ms \
          --filterbank_nchannels=$mel \
          --dct_ncoefficients=$dct \
          --model_architecture=$architecture \
          --model_parameters='$model_parameters' \
          --representation=$representation \
          --optimizer=$optimizer \
          --batch_size=$mini_batch"

          #--subsample_label='mel-pulse' \
          #--subsample_skip=1 \

    cmd=${cmd}" $expr $redirect $logdir/xvalidate_${model}.log & "
done

cmd=${cmd}"wait; sync; date"
echo $cmd

eval "$cmd"
