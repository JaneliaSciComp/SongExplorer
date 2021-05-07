#!/bin/bash

# train several networks withholding different subsets of the recordings to test upon

# generalize.sh <context-ms> <shiftby-ms> <optimizer> <learning-rate> <model-architecture> <model-parameters-json> <logdir> <path-to-groundtruth> <label1>,<label2>,...,<labelN> <kinds-to-use> <nsteps> <restore-from> <save-and-validate-period> <mini-batch> <testing-files> <audio-tic-rate> <audio-nchannels> <batch-seed> <weights-seed> <ioffset> <subset1> [<subset2> [<subset3>]...]

# e.g.
# $SONGEXPLORER_BIN generalize.sh 204.8 0.0 Adam 0.0002 convolutional '{"representation":"waveform", "window_ms":6.4, "stride_ms":1.6, "mel_dct":"7,7", "dropout":0.5, "kernel_sizes":5,128", last_conv_width":130, "nfeatures":"256,256", "dilate_after_layer":65535, "stride_after_layer":65535, "connection_type":"plain"}' `pwd`/leave-one-out `pwd`/groundtruth-data mel-pulse,mel-sine,ambient,other annotated 50 '' 10 32 "" 5000 1 -1 -1 3 20161207T102314_ch1_p1.wav,20161207T102314_ch1_p2.wav,20161207T102314_ch1_p3.wav PS_20130625111709_ch3_p1.wav,PS_20130625111709_ch3_p2.wav,PS_20130625111709_ch3_p3.wav

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
ioffset=${20}

shift 20

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

echo SongExplorer version: $(cat $DIR/../VERSION.txt)

mkdir -p $logdir

if [ -z "$restore_from" ] ; then
  redirect='&>'
  start_checkpoint=
else
  redirect='&>>'
  start_checkpoint=$logdir/generalize_MODEL/ckpt-$restore_from
fi

cmd="date; hostname; echo $CUDA_VISIBLE_DEVICES; nvidia-smi; "
    
isubset=1
while (( $# > 0 )) ; do
    model=$((ioffset+isubset))w
    expr="/usr/bin/python3 $DIR/speech_commands_custom/train.py \
          --data_dir=$data_dir \
          --labels_touse=$labels_touse \
          --kinds_touse=$kinds_touse \
          --how_many_training_steps=$nsteps \
          --start_checkpoint=${start_checkpoint/MODEL/$model} \
          --save_step_period=$save_and_validate_period \
          --validate_step_period=$save_and_validate_period \
          --train_dir=$logdir/generalize_$model \
          --summaries_dir=$logdir/summaries_$model \
          --audio_tic_rate=$audio_tic_rate \
          --nchannels=$audio_nchannels \
          --context_ms=$context_ms \
          --learning_rate=$learning_rate \
          --random_seed_batch=$batch_seed \
          --random_seed_weights=$weights_seed \
          --validation_files=$1 \
          --validation_percentage=0 \
          --validation_offset_percentage=0 \
          --testing_files=$testing_files \
          --shiftby_ms=$shiftby_ms \
          --model_architecture=$architecture \
          --model_parameters='$model_parameters' \
          --optimizer=$optimizer \
          --batch_size=$mini_batch"

          #--subsample_label=mel-notpulse,mel-pulse,mel-time \
          #--subsample_skip=2048,2048,256 \

    cmd=${cmd}" $expr $redirect $logdir/generalize_${model}.log & "
    shift 1
    (( isubset++ ))
done

cmd=${cmd}"wait; sync; date"
echo $cmd

eval "$cmd"
