#!/usr/bin/env bash

# test that the shiftby hyperparameter works

#${SONGEXPLORER_BIN/-B/-B /tmp:/opt/songexplorer/test/scratch -B} bash -c "test/shiftby.sh"

repo_path=$(dirname $(dirname $(readlink -f $(which songexplorer))))

mkdir -p $repo_path/test/scratch/shiftby
cp $repo_path/configuration.py $repo_path/test/scratch/shiftby

source $repo_path/test/scratch/shiftby/configuration.py

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
model_parameters='{"representation":"waveform","window_ms":3.2,"stride_ms":0.8,"mel_dct":"3,3","range_hz":"","dropout_kind":"unit","dropout_rate":"50","augment_volume":"1,1","augment_noise":"0,0","normalization":"none","kernel_sizes":"3x3,32","nconvlayers":"2","denselayers":"","nfeatures":"8,8","dilate_time":"","dilate_freq":"","stride_time":"2", "stride_freq":"","pool_kind":"none","pool_size":"","connection_type":"plain"}'
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

cmd="train \
     --context_ms=$context_ms \
     --shiftby_ms=$shiftby_ms \
     --optimizer=$optimizer \
     --learning_rate=$learning_rate \
     --audio_read_plugin=$audio_read_plugin \
     --audio_read_plugin_kwargs=$audio_read_plugin_kwargs \
     --video_read_plugin=$video_read_plugin \
     --video_read_plugin_kwargs=$video_read_plugin_kwargs \
     --video_findfile_plugin=$video_findfile_plugin \
     --video_bkg_frames=$video_bkg_frames \
     --data_loader_queuesize=$data_loader_queuesize \
     --data_loader_maxprocs=$data_loader_maxprocs \
     --model_architecture=$architecture \
     --model_parameters='$model_parameters' \
     --logdir=$logdir \
     --data_dir=$data_dir \
     --labels_touse=$labels_touse \
     --kinds_touse=$kinds_touse \
     --nsteps=$nsteps \
     --restore_from='$restore_from' \
     --save_and_test_period=$save_and_test_period \
     --validation_percentage=$validation_percentage \
     --mini_batch=$mini_batch \
     --testing_files='$testing_files' \
     --audio_tic_rate=$audio_tic_rate \
     --audio_nchannels=$audio_nchannels \
     --video_frame_rate=$video_frame_rate \
     --video_frame_width=$video_frame_width \
     --video_frame_height=$video_frame_height \
     --video_channels=$video_channels \
     --batch_seed=$batch_seed \
     --weights_seed=$weights_seed \
     --deterministic=$deterministic \
     --ireplicates=$ireplicates \
     --save_fingerprints=0"
echo $cmd &> $logdir/train1.log
eval $cmd &> $logdir/train1.log

shiftby_ms=51.2
logdir=$repo_path/test/scratch/shiftby/shiftby-$shiftby_ms
mkdir $logdir

cmd="train \
     --context_ms=$context_ms \
     --shiftby_ms=$shiftby_ms \
     --optimizer=$optimizer \
     --learning_rate=$learning_rate \
     --audio_read_plugin=$audio_read_plugin \
     --audio_read_plugin_kwargs=$audio_read_plugin_kwargs \
     --video_read_plugin=$video_read_plugin \
     --video_read_plugin_kwargs=$video_read_plugin_kwargs \
     --video_findfile_plugin=$video_findfile_plugin \
     --video_bkg_frames=$video_bkg_frames \
     --data_loader_queuesize=$data_loader_queuesize \
     --data_loader_maxprocs=$data_loader_maxprocs \
     --architecture=$architecture \
     --model_parameters='$model_parameters' \
     --logdir=$logdir \
     --data_dir=$data_dir \
     --labels_touse=$labels_touse \
     --kinds_touse=$kinds_touse \
     --nsteps=$nsteps \
     --restore_from='$restore_from' \
     --save_and_test_period=$save_and_test_period \
     --validation_percentage=$validation_percentage \
     --mini_batch=$mini_batch \
     --testing_files='$testing_files' \
     --audio_tic_rate=$audio_tic_rate \
     --audio_nchannels=$audio_nchannels \
     --video_frame_rate=$video_frame_rate \
     --video_frame_width=$video_frame_width \
     --video_frame_height=$video_frame_height \
     --video_channels=$video_channels \
     --batch_seed=$batch_seed \
     --weights_seed=$weights_seed \
     --deterministic=$deterministic \
     --ireplicates=$ireplicates \
     --save_fingerprints=0"
echo $cmd &> $logdir/train1.log
eval $cmd &> $logdir/train1.log
