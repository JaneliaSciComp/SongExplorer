#!/bin/bash

# test that the shiftby hyperparameter works

#${SONGEXPLORER_BIN/-B/-B /tmp:/opt/songexplorer/test/scratch -B} bash -c "test/shiftby.sh"

repo_path=$(dirname $(dirname $(which detect.sh)))

mkdir -p $repo_path/test/scratch/shiftby
cp $repo_path/configuration.pysh $repo_path/test/scratch/shiftby

source $repo_path/test/scratch/shiftby/configuration.pysh

mkdir -p $repo_path/test/scratch/shiftby/groundtruth-data/round1
cp $repo_path/data/PS_20130625111709_ch3.wav \
   $repo_path/test/scratch/shiftby/groundtruth-data/round1
cp $repo_path/data/PS_20130625111709_ch3-annotated-person1.csv \
   $repo_path/test/scratch/shiftby/groundtruth-data/round1

context_ms=204.8
shiftby_ms=0.0
representation=waveform
window_ms=6.4
mel=7
dct=7
stride_ms=1.6
optimizer=adam
learning_rate=0.000001
architecture=convolutional
model_parameters='{"dropout": "0.5", "kernel_sizes": "5,3,128", "nlayers": "3", "nfeatures": "8,8,8", "dilate_after_layer": "65535", "stride_after_layer": "0", "connection_type": "plain"}'
logdir=$repo_path/test/scratch/shiftby/shiftby-$shiftby_ms
data_dir=$repo_path/test/scratch/shiftby/groundtruth-data
wanted_words=mel-pulse,mel-sine,ambient
labels_touse=annotated
nsteps=10
save_and_test_interval=5
validation_percentage=20
restore_from=''
mini_batch=32
testing_files=''
batch_seed=1
weights_seed=1
ireplicates=1
mkdir $logdir

train.sh \
      $context_ms $shiftby_ms $representation $window_ms $stride_ms $mel $dct \
      $optimizer $learning_rate \
      $architecture "$model_parameters" \
      $logdir $data_dir $wanted_words $labels_touse \
      $nsteps "$restore_from" $save_and_test_interval $validation_percentage \
      $mini_batch "$testing_files" \
      $audio_tic_rate $audio_nchannels \
      $batch_seed $weights_seed $ireplicates \
      True \
      &> $logdir/train1.log

shiftby_ms=51.2
logdir=$repo_path/test/scratch/shiftby/shiftby-$shiftby_ms
mkdir $logdir

train.sh \
      $context_ms $shiftby_ms $representation $window_ms $stride_ms $mel $dct \
      $optimizer $learning_rate \
      $architecture "$model_parameters" \
      $logdir $data_dir $wanted_words $labels_touse \
      $nsteps "$restore_from" $save_and_test_interval $validation_percentage \
      $mini_batch "$testing_files" \
      $audio_tic_rate $audio_nchannels \
      $batch_seed $weights_seed $ireplicates \
      True \
      &> $logdir/train1.log
