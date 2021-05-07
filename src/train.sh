#!/bin/bash

# train a neural network with the annotations

# train.sh <context-ms> <shiftby-ms> <optimizer> <learning_rate> <model-architecture> <model-parameters-json> <logdir> <path-to-groundtruth> <label1>,<label2>,...,<labelN> <kinds-to-use> <nsteps> <restore-from> <save-and-validate-period> <validation-percentage> <mini-batch> <testing-files> <audio-tic-rate> <audio-nchannels> <batch-seed> <weights-seed> <ireplicates>

# e.g.
# $SONGEXPLORER_BIN train.sh 204.8 0.0 Adam 0.0002 convolutional '{"representation":"waveform", "window_ms":6.4, "stride_ms":1.6, "mel_dct":"7,7", "dropout":0.5, "kernel_sizes":5,128", last_conv_width":130, "nfeatures":"256,256", "dilate_after_layer":65535, "stride_after_layer":65535, "connection_type":"plain"}' `pwd`/trained-classifier `pwd`/groundtruth-data mel-sine,mel-pulse,ambient,other annotated 50 '' 10 40 32 "" 5000 1 -1 -1 1,2,3,4

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
validation_percentage=${14}
mini_batch=${15}
testing_files=${16}
audio_tic_rate=${17}
audio_nchannels=${18}
batch_seed=${19}
weights_seed=${20}
ireplicates=${21}

if (( "$#" == 21 )) ; then
  save_fingerprints=False
else
  save_fingerprints=${22}
fi

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

echo SongExplorer version: $(cat $DIR/../VERSION.txt)

mkdir -p $logdir

if [ -z "$restore_from" ] ; then
  redirect='&>'
  start_checkpoint=
else
  redirect='&>>'
  start_checkpoint=$logdir/train_MODEL/ckpt-$restore_from
fi

cmd="date; hostname; echo $CUDA_VISIBLE_DEVICES; nvidia-smi; "
    
ireplicates=${ireplicates},
while [[ $ireplicates =~ .*,.* ]] ; do
    ireplicate=${ireplicates%%,*}
    ireplicates=${ireplicates#*,}

    expr="/usr/bin/python3 $DIR/speech_commands_custom/train.py \
          --data_dir=$data_dir \
          --labels_touse=$labels_touse \
          --kinds_touse=$kinds_touse \
          --how_many_training_steps=$nsteps \
          --start_checkpoint=${start_checkpoint/MODEL/${ireplicate}r} \
          --save_step_period=$save_and_validate_period \
          --validate_step_period=$save_and_validate_period \
          --train_dir=$logdir/train_${ireplicate}r \
          --summaries_dir=$logdir/summaries_${ireplicate}r \
          --audio_tic_rate=$audio_tic_rate \
          --nchannels=$audio_nchannels \
          --random_seed_batch=$batch_seed \
          --random_seed_weights=$weights_seed \
          --validation_percentage=$validation_percentage \
          --validation_offset_percentage=0.0 \
          --testing_percentage=0.0 \
          --testing_files=$testing_files \
          --shiftby_ms=$shiftby_ms \
          --context_ms=$context_ms \
          --optimizer=$optimizer \
          --learning_rate=${learning_rate} \
          --model_architecture=$architecture \
          --model_parameters='$model_parameters' \
          --save_fingerprints=$save_fingerprints \
          --batch_size=$mini_batch"

          #--subsample_label=mel-pulse,mel-notpulse \
          #--subsample_skip=4096 \

          #--partition_label=mel-pulse,mel-notpulse \
          #--partition_n=4 \
          #--partition_training_files=PS_20130625111709_ch10.wav,PS_20130625111709_ch3.wav,PS_20130625155828_ch10.wav,PS_20130625155828_ch11.wav,PS_20130625155828_ch3.wav,PS_20130625155828_ch7.wav,PS_20130625155828_ch8.wav,PS_20130628144304_ch14.wav,PS_20130628144304_ch16.wav,PS_20130628144304_ch2.wav,PS_20130628144304_ch8.wav,PS_20130628165930_ch11.wav,PS_20130702114557_ch1.wav,PS_20130702114557_ch13.wav,PS_20130702114557_ch14.wav,PS_20130702144748_ch15.wav \
          #--partition_validation_files=PS_20130625111709_ch7.wav,PS_20130625155828_ch6.wav,PS_20130628144304_ch15.wav,PS_20130702114557_ch10.wav \

    cmd=${cmd}" $expr $redirect $logdir/train_${ireplicate}r.log & "
done

cmd=${cmd}"wait; sync; date"
echo $cmd

eval "$cmd"
