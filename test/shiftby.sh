#!/bin/bash

# test that the shiftby hyperparameter works

#${SONGEXPLORER_BIN/-B/-B /tmp:/opt/songexplorer/test/scratch -B} bash -c "test/shiftby.sh"

repo_path=$(dirname $(dirname $(which detect.py)))

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
optimizer=Adam
learning_rate=0.000001
architecture=convolutional
model_parameters='{"representation":"waveform", "window_ms":3.2, "stride_ms":0.8, "mel_dct":"3,3", "dropout": "0.5", "augment_volume": "1,1", "augment_noise": "0,0", "kernel_sizes": "3,32", "nlayers": "2", "nfeatures": "8,8", "dilate_after_layer": "65535", "stride_after_layer": "2", "connection_type": "plain"}'
logdir=$repo_path/test/scratch/shiftby/shiftby-$shiftby_ms
data_dir=$repo_path/test/scratch/shiftby/groundtruth-data
labels_touse=mel-pulse,mel-sine,ambient
kinds_touse=annotated
nsteps=10
save_and_test_period=5
validation_percentage=20
restore_from=''
mini_batch=32
testing_files=''
batch_seed=1
weights_seed=1
ireplicates=1
mkdir $logdir

train.py \
      $context_ms $shiftby_ms \
      $optimizer $learning_rate \
      $architecture "$model_parameters" \
      $logdir $data_dir $labels_touse $kinds_touse \
      $nsteps "$restore_from" $save_and_test_period $validation_percentage \
      $mini_batch "$testing_files" \
      $audio_tic_rate $audio_nchannels \
      $batch_seed $weights_seed $deterministic $ireplicates \
      True \
      &> $logdir/train1.log

shiftby_ms=51.2
logdir=$repo_path/test/scratch/shiftby/shiftby-$shiftby_ms
mkdir $logdir

train.py \
      $context_ms $shiftby_ms \
      $optimizer $learning_rate \
      $architecture "$model_parameters" \
      $logdir $data_dir $labels_touse $kinds_touse \
      $nsteps "$restore_from" $save_and_test_period $validation_percentage \
      $mini_batch "$testing_files" \
      $audio_tic_rate $audio_nchannels \
      $batch_seed $weights_seed $deterministic $ireplicates \
      True \
      &> $logdir/train1.log
