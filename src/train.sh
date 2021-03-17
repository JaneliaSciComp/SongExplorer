#!/bin/bash

# train a neural network with the annotations

# train.sh <model-architecture> <context-ms> <shiftby-ms> <representation> <window-ms> <mel> <dct> <stride-ms> <dropout> <optimizer> <learning-rate> <kernel-sizes> <last-conv-width> <nfeatures> <dilate-after-layer> <stride-after-layer> <connection-type> <logdir> <path-to-groundtruth> <word1>,<word2>,...,<wordN> <label-types> <nsteps> <restore-from> <save-and-test-interval> <validation-percentage> <mini-batch> <testing-files> <audio-tic-rate> <audio-nchannels> <batch-seed> <weights-seed> <ireplicates>

# e.g.
# $SONGEXPLORER_BIN train.sh convolutional 204.8 0.0 waveform 6.4 7 7 1.6 0.5 adam 0.0002 5,3,3 130 256,256,256 65535 65535 plain `pwd`/trained-classifier `pwd`/groundtruth-data mel-sine,mel-pulse,ambient,other annotated 50 '' 10 40 32 "" 5000 1 -1 -1 1,2,3,4

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
validation_percentage=${25}
mini_batch=${26}
testing_files=${27}
audio_tic_rate=${28}
audio_nchannels=${29}
batch_seed=${30}
weights_seed=${31}
ireplicates=${32}

if (( "$#" == 32 )) ; then
  save_fingerprints=False
else
  save_fingerprints=${33}
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
          --clip_duration_ms=$context_ms \
          --window_size_ms=$window_ms \
          --window_stride_ms=$stride_ms \
          --learning_rate=${learning_rate} \
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
