#!/bin/bash

# train several networks on different subsets of the annotations

# xvalidate.sh <context-ms> <shiftby-ms> <optimizer> <learning-rate> <model-architecture> <model-parameters-json> <logdir> <path-to-groundtruth> <label1>,<label2>,...,<labelN> <kinds-to-use> <nsteps> <restore-from> <save-and-validate-period> <mini-batch> <testing-files> <audio-tic-rate> <audio-nchannels> <batch-seed> <weights-seed> <kfold> <ifolds>

# e.g.
# $SONGEXPLORER_BIN xvalidate.sh 204.8 0.0 Adam 0.0002 convolutional '{"representation":"waveform", "window_ms":6.4, "stride_ms":1.6, "mel_dct":"7,7", "dropout":0.5, "kernel_sizes":5,128", last_conv_width":130, "nfeatures":"256,256", "dilate_after_layer":65535, "stride_after_layer":65535, "connection_type":"plain"}' `pwd`/cross-validate `pwd`/groundtruth-data mel-pulse,mel-sine,ambient,other annotated 50 '' 10 32 "" 5000 1 -1 -1 8 1,2

context_ms=$1
shiftby_ms=$2
optimizer=$3
learning_rate=$4
architecture=$5
model_parameters=$6
logdir=$7
data_dir=$8
labels_touse=$9
kinds_touse=${10}
nsteps=${11}
restore_from=${12}
save_and_validate_period=${13}
mini_batch=${14}
testing_files=${15}
audio_tic_rate=${16}
audio_nchannels=${17}
batch_seed=${18}
weights_seed=${19}
kfold=${20}
ifolds=${21}

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

echo SongExplorer version: $(cat $DIR/../VERSION.txt)

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
          --learning_rate=$learning_rate \
          --random_seed_batch=$batch_seed \
          --random_seed_weights=$weights_seed \
          --validation_percentage=$kpercent \
          --validation_offset_percentage=$koffset \
          --testing_files=$testing_files \
          --shiftby_ms=$shiftby_ms \
          --model_architecture=$architecture \
          --model_parameters='$model_parameters' \
          --optimizer=$optimizer \
          --batch_size=$mini_batch"

          #--subsample_label='mel-pulse' \
          #--subsample_skip=1 \

    cmd=${cmd}" $expr $redirect $logdir/xvalidate_${model}.log & "
done

cmd=${cmd}"wait; sync; date"
echo $cmd

eval "$cmd"
