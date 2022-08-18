#!/usr/bin/env bash

# test that the shiftby hyperparameter works

#${SONGEXPLORER_BIN/-B/-B /tmp:/opt/songexplorer/test/scratch -B} bash -c "test/shiftby.sh"

repo_path=$(dirname $(dirname $(readlink -f $(which songexplorer))))

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
model_parameters='{"representation":"waveform","window_ms":3.2,"stride_ms":0.8,"mel_dct":"3,3","dropout_kind":"unit","dropout_rate":"50","augment_volume":"1,1","augment_noise":"0,0","normalization":"none","kernel_sizes":"3x3,32","nconvlayers":"2","denselayers":"","nfeatures":"8,8","dilate_after_layer":"256,256","stride_after_layer":"2,256","connection_type":"plain"}'
logdir=$repo_path/test/scratch/shiftby/shiftby-$shiftby_ms
data_dir=$repo_path/test/scratch/shiftby/groundtruth-data
labels_touse=mel-pulse,mel-sine,ambient
kinds_touse=annotated
nsteps=10
save_and_test_period=5
validation_percentage=20
restore_from=
mini_batch=32
testing_files=
batch_seed=1
weights_seed=1
ireplicates=1
mkdir $logdir

cmd="train.py \
      $context_ms $shiftby_ms \
      $optimizer $learning_rate \
      $audio_read_plugin $audio_read_plugin_kwargs \
      $video_read_plugin $video_read_plugin_kwargs \
      $video_findfile_plugin $video_bkg_frames \
      $data_loader_queuesize $data_loader_maxprocs \
      $architecture '$model_parameters' \
      $logdir $data_dir $labels_touse $kinds_touse \
      $nsteps '$restore_from' $save_and_test_period $validation_percentage \
      $mini_batch '$testing_files' \
      $audio_tic_rate $audio_nchannels \
      $video_frame_rate $video_frame_width $video_frame_height $video_channels \
      $batch_seed $weights_seed $deterministic $ireplicates \
      True"
echo $cmd &> $logdir/train1.log
eval $cmd &> $logdir/train1.log

shiftby_ms=51.2
logdir=$repo_path/test/scratch/shiftby/shiftby-$shiftby_ms
mkdir $logdir

cmd="train.py \
      $context_ms $shiftby_ms \
      $optimizer $learning_rate \
      $audio_read_plugin $audio_read_plugin_kwargs \
      $video_read_plugin $video_read_plugin_kwargs \
      $video_findfile_plugin $video_bkg_frames \
      $data_loader_queuesize $data_loader_maxprocs \
      $architecture '$model_parameters' \
      $logdir $data_dir $labels_touse $kinds_touse \
      $nsteps '$restore_from' $save_and_test_period $validation_percentage \
      $mini_batch '$testing_files' \
      $audio_tic_rate $audio_nchannels \
      $video_frame_rate $video_frame_width $video_frame_height $video_channels \
      $batch_seed $weights_seed $deterministic $ireplicates \
      True"
echo $cmd &> $logdir/train1.log
eval $cmd &> $logdir/train1.log
