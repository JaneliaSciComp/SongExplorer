#!/bin/bash

# train a neural network with the annotations

# train.sh <context-ms> <shiftby-ms> <representation> <window_ms> <stride_ms> <mel> <dct> <optimizer> <learning_rate> <model-architecture> <model-parameters-json> <logdir> <path-to-groundtruth> <word1>,<word2>,...,<wordN> <label-types> <nsteps> <restore-from> <save-and-test-interval> <validation-percentage> <mini-batch> <testing-files> <audio-tic-rate> <audio-nchannels> <batch-seed> <weights-seed> <ireplicates>

# e.g.
# $SONGEXPLORER_BIN train.sh 204.8 0.0 waveform 6.4 1.6 7 7 adam 0.0002 convolutional '{"dropout":0.5, "kernel_sizes":5,3,3", last_conv_width":130, "nfeatures":"256,256,256", "dilate_after_layer":65535, "stride_after_layer":65535, "connection_type":"plain"}' `pwd`/trained-classifier `pwd`/groundtruth-data mel-sine,mel-pulse,ambient,other annotated 50 '' 10 40 32 "" 5000 1 -1 -1 1,2,3,4

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
wanted_words=${14}
labels_touse=${15}
nsteps=${16}
restore_from=${17}
save_and_test_interval=${18}
validation_percentage=${19}
mini_batch=${20}
testing_files=${21}
audio_tic_rate=${22}
audio_nchannels=${23}
batch_seed=${24}
weights_seed=${25}
ireplicates=${26}

if (( "$#" == 26 )) ; then
  save_fingerprints=False
else
  save_fingerprints=${27}
fi

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

mkdir -p $logdir

if [ -z "$restore_from" ] ; then
  redirect='&>'
  start_checkpoint=
else
  redirect='&>>'
  start_checkpoint=$logdir/train_MODEL/${architecture}.ckpt-$restore_from
fi

cmd="date; hostname; echo $CUDA_VISIBLE_DEVICES; nvidia-smi; "
    
ireplicates=${ireplicates},
while [[ $ireplicates =~ .*,.* ]] ; do
    ireplicate=${ireplicates%%,*}
    ireplicates=${ireplicates#*,}

    expr="/usr/bin/python3 $DIR/speech_commands_custom/train.py \
          --data_url= --data_dir=$data_dir \
          --wanted_words=$wanted_words \
          --labels_touse=$labels_touse \
          --how_many_training_steps=$nsteps \
          --start_checkpoint=${start_checkpoint/MODEL/${ireplicate}r} \
          --save_step_interval=$save_and_test_interval \
          --eval_step_interval=$save_and_test_interval \
          --train_dir=$logdir/train_${ireplicate}r \
          --summaries_dir=$logdir/summaries_${ireplicate}r \
          --sample_rate=$audio_tic_rate \
          --nchannels=$audio_nchannels \
          --random_seed_batch=$batch_seed \
          --random_seed_weights=$weights_seed \
          --background_frequency=0.0 \
          --silence_percentage=0.0 \
          --unknown_percentage=0.0 \
          --validation_percentage=$validation_percentage \
          --validation_offset_percentage=0.0 \
          --testing_percentage=0.0 \
          --testing_files=$testing_files \
          --time_shift_ms=$shiftby_ms \
          --time_shift_random False \
          --clip_duration_ms=$context_ms \
          --representation=$representation \
          --window_size_ms=$window_ms \
          --window_stride_ms=$stride_ms \
          --filterbank_channel_count=$mel \
          --dct_coefficient_count=$dct \
          --optimizer=$optimizer \
          --learning_rate=${learning_rate} \
          --model_architecture=$architecture \
          --model_parameters='$model_parameters' \
          --save_fingerprints=$save_fingerprints \
          --batch_size=$mini_batch"

          #--subsample_word=mel-pulse,mel-notpulse \
          #--subsample_skip=4096 \

          #--partition_word=mel-pulse,mel-notpulse \
          #--partition_n=4 \
          #--partition_training_files=PS_20130625111709_ch10.wav,PS_20130625111709_ch3.wav,PS_20130625155828_ch10.wav,PS_20130625155828_ch11.wav,PS_20130625155828_ch3.wav,PS_20130625155828_ch7.wav,PS_20130625155828_ch8.wav,PS_20130628144304_ch14.wav,PS_20130628144304_ch16.wav,PS_20130628144304_ch2.wav,PS_20130628144304_ch8.wav,PS_20130628165930_ch11.wav,PS_20130702114557_ch1.wav,PS_20130702114557_ch13.wav,PS_20130702114557_ch14.wav,PS_20130702144748_ch15.wav \
          #--partition_validation_files=PS_20130625111709_ch7.wav,PS_20130625155828_ch6.wav,PS_20130628144304_ch15.wav,PS_20130702114557_ch10.wav \

    cmd=${cmd}" $expr $redirect $logdir/train_${ireplicate}r.log & "
done

cmd=${cmd}"wait; sync; date"
echo $cmd

eval "$cmd"
