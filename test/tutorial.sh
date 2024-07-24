#!/usr/bin/env bash

# recapitulate the tutorial via the shell interface

check_file_exists() {
  if [[ ! -e $1 ]] ; then
    echo ERROR: $1 is missing
    return 1
  fi
  return 0; }

count_lines_with_label() {
  check_file_exists $1 || return
  local count=$(grep $2 $1 | wc -l)
  (( $count == $3 )) && return
  echo $4: $1 has $count $2 when it should have $3
  if [ "$4" == "WARNING" ]; then echo $4: it is normal for this to be close but not exact; fi; }

count_lines() {
  check_file_exists $1 || return
  local count=$(cat $1 | wc -l)
  (( $count == $2 )) && return
  echo ERROR: $1 has $count lines when it should have $2; }

testdir="$( cd "$( dirname "$(readlink -f ${BASH_SOURCE[0]})" )" >/dev/null 2>&1 && pwd )"
repo_path=$(dirname $testdir)
bindir=$(dirname $repo_path)
srcdir=${repo_path}/src

PATH=$PATH:$bindir

mkdir -p $repo_path/test/scratch/tutorial-sh
cp $repo_path/configuration.py $repo_path/test/scratch/tutorial-sh

source $repo_path/test/scratch/tutorial-sh/configuration.py
deterministic=1

mkdir -p $repo_path/test/scratch/tutorial-sh/groundtruth-data/round1
cp $repo_path/data/PS_20130625111709_ch3.wav \
   $repo_path/test/scratch/tutorial-sh/groundtruth-data/round1

time_units=ms
freq_units=Hz
time_scale=0.001
freq_scale=1

wavpath_noext=$repo_path/test/scratch/tutorial-sh/groundtruth-data/round1/PS_20130625111709_ch3
detect_parameters='{"time_sigma":"9,4","time_smooth":"6.4","frequency_n":"25.6","frequency_nw":"4","frequency_p":"0.1,1.0","frequency_range":"0-","frequency_smooth":"25.6","time_sigma_robust":"median"}'
cmd="${srcdir}/${detect_plugin}.py \
      --filename=${wavpath_noext}.wav \
      --parameters='$detect_parameters' \
      --time_units=$time_units \
      --freq_units=$freq_units \
      --time_scale=$time_scale \
      --freq_scale=$freq_scale \
      --audio_tic_rate=$audio_tic_rate \
      --audio_nchannels=$audio_nchannels \
      --audio_read_plugin=$audio_read_plugin \
      --audio_read_plugin_kwargs=$audio_read_plugin_kwargs"
echo $cmd >> ${wavpath_noext}-detect.log 2>&1
eval $cmd >> ${wavpath_noext}-detect.log 2>&1

check_file_exists ${wavpath_noext}-detect.log
check_file_exists ${wavpath_noext}-detected.csv
count_lines_with_label ${wavpath_noext}-detected.csv time 536 ERROR
count_lines_with_label ${wavpath_noext}-detected.csv frequency 45 ERROR
count_lines_with_label ${wavpath_noext}-detected.csv neither 1635 ERROR

cp $repo_path/data/PS_20130625111709_ch3-annotated-person1.csv \
   $repo_path/test/scratch/tutorial-sh/groundtruth-data/round1

context=204.8
shiftby=0.0
optimizer=Adam
learning_rate=0.0002
architecture=convolutional
model_parameters='{"representation":"mel-cepstrum","window":"6.4","stride":"1.6","mel_dct":"7,7","range":"","dropout_kind":"unit","dropout_rate":"50","augment_volume":"1,1","augment_noise":"0,0","normalization":"none","kernel_sizes":"5x5,3","nconvlayers":"2","denselayers":"","nfeatures":"64,64","stride_time":"","stride_freq":"","dilate_time":"","dilate_freq":"","pool_kind":"none","pool_size":"","connection_type":"plain"}'
logdir=$repo_path/test/scratch/tutorial-sh/trained-classifier1
data_dir=$repo_path/test/scratch/tutorial-sh/groundtruth-data
labels_touse=mel-pulse,mel-sine,ambient
kinds_touse=annotated
nsteps=300
restore_from=
save_and_validate_period=30
validation_percentage=40
mini_batch=32
testing_files=
batch_seed=1
weights_seed=1
ireplicates=1
loss=exclusive
overlapped_prefix=not_
mkdir $logdir
cmd="${srcdir}/train \
     --context=$context \
     --shiftby=$shiftby \
     --optimizer=$optimizer \
     --loss=$loss \
     --overlapped_prefix=$overlapped_prefix \
     --learning_rate=$learning_rate  \
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
     --save_and_validate_period=$save_and_validate_period \
     --validation_percentage=$validation_percentage \
     --mini_batch=$mini_batch \
     --testing_files='$testing_files' \
     --time_units=$time_units \
     --freq_units=$freq_units \
     --time_scale=$time_scale \
     --freq_scale=$freq_scale \
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
     --igpu="
echo $cmd >> $logdir/train1.log 2>&1
eval $cmd >> $logdir/train1.log 2>&1

check_file_exists $logdir/train1.log
check_file_exists $logdir/train_1r.log
check_file_exists $logdir/train_1r/ckpt-$nsteps.index
check_file_exists $logdir/train_1r/logits.validation.ckpt-$nsteps.npz

precision_recall_ratios=0.5,1.0,2.0
cmd="${srcdir}/accuracy \
     --logdir=$logdir \
     --loss=$loss \
     --overlapped_prefix=$overlapped_prefix \
     --error_ratios=$precision_recall_ratios \
     --nprobabilities=$nprobabilities \
     --parallelize=$accuracy_parallelize"
echo $cmd >> $logdir/accuracy.log 2>&1
eval $cmd >> $logdir/accuracy.log 2>&1

check_file_exists $logdir/accuracy.log
check_file_exists $logdir/precision-recall.pdf
check_file_exists $logdir/confusion-matrix.pdf
check_file_exists $logdir/train_1r/precision-recall.ckpt-$nsteps.pdf
check_file_exists $logdir/train_1r/probability-density.ckpt-$nsteps.pdf
check_file_exists $logdir/train_1r/thresholds.ckpt-$nsteps.csv
check_file_exists $logdir/train_1r/confusion-matrix.ckpt-$nsteps.pdf
check_file_exists $logdir/train-validation-loss.pdf
check_file_exists $logdir/P-R-F1-average.pdf
check_file_exists $logdir/P-R-F1-label.pdf
check_file_exists $logdir/P-R-F1-model.pdf
check_file_exists $logdir/PvR.pdf

cmd="${srcdir}/freeze \
      --context=$context \
      --loss=$loss \
      --model_architecture=$architecture \
      --model_parameters='$model_parameters' \
      --start_checkpoint=${logdir}/train_${ireplicates}r/ckpt-$nsteps \
      --output_file=${logdir}/train_${ireplicates}r/frozen-graph.ckpt-${nsteps}.pb \
      --labels_touse=$labels_touse \
      --parallelize=$classify_parallelize \
      --time_units=$time_units \
      --freq_units=$freq_units \
      --time_scale=$time_scale \
      --freq_scale=$freq_scale \
      --audio_tic_rate=$audio_tic_rate \
      --audio_nchannels=$audio_nchannels \
      --video_frame_rate=$video_frame_rate \
      --video_frame_height=$video_frame_height \
      --video_frame_width=$video_frame_width \
      --video_channels=$video_channels \
      --igpu="
echo $cmd >> $logdir/train_${ireplicates}r/freeze.ckpt-${nsteps}.log 2>&1
eval $cmd >> $logdir/train_${ireplicates}r/freeze.ckpt-${nsteps}.log 2>&1

check_file_exists $logdir/train_${ireplicates}r/freeze.ckpt-${nsteps}.log
check_file_exists $logdir/train_${ireplicates}r/frozen-graph.ckpt-${nsteps}.pb

mkdir $repo_path/test/scratch/tutorial-sh/groundtruth-data/round2
cp $repo_path/data/20161207T102314_ch1.wav \
   $repo_path/test/scratch/tutorial-sh/groundtruth-data/round2

wavpath_noext=$repo_path/test/scratch/tutorial-sh/groundtruth-data/round2/20161207T102314_ch1
cmd="${srcdir}/classify \
      --context=$context \
      --loss=$loss \
      --shiftby=$shiftby \
      --audio_read_plugin=$audio_read_plugin \
      --audio_read_plugin_kwargs=$audio_read_plugin_kwargs \
      --video_read_plugin=$video_read_plugin \
      --video_read_plugin_kwargs=$video_read_plugin_kwargs \
      --video_findfile_plugin=$video_findfile_plugin \
      --video_bkg_frames=$video_bkg_frames \
      --model=$logdir/train_${ireplicates}r/frozen-graph.ckpt-${nsteps}.pb \
      --model_labels=$logdir/train_${ireplicates}r/labels.txt \
      --wav=${wavpath_noext}.wav \
      --parallelize=$classify_parallelize \
      --time_scale=$time_scale \
      --audio_tic_rate=$audio_tic_rate \
      --audio_nchannels=$audio_nchannels \
      --video_frame_rate=$video_frame_rate \
      --video_frame_height=$video_frame_height \
      --video_frame_width=$video_frame_width \
      --video_channels=$video_channels \
      --deterministic=$deterministic \
      --labels= \
      --prevalences= \
      --igpu="
echo $cmd >> ${wavpath_noext}-classify.log 2>&1
eval $cmd >> ${wavpath_noext}-classify.log 2>&1

check_file_exists ${wavpath_noext}-classify.log

for label in $(echo $labels_touse | sed "s/,/ /g") ; do
  check_file_exists ${wavpath_noext}-${label}.wav
done

cmd="${srcdir}/ethogram \
      $logdir train_${ireplicates}r thresholds.ckpt-${nsteps}.csv \
      ${wavpath_noext}.wav $audio_tic_rate"
echo $cmd >> ${wavpath_noext}-ethogram.log 2>&1
eval $cmd >> ${wavpath_noext}-ethogram.log 2>&1

check_file_exists ${wavpath_noext}-ethogram.log
for pr in $(echo $precision_recall_ratios | sed "s/,/ /g") ; do
  check_file_exists ${wavpath_noext}-predicted-${pr}pr.csv
done
count_lines_with_label ${wavpath_noext}-predicted-1.0pr.csv mel-pulse 510 WARNING
count_lines_with_label ${wavpath_noext}-predicted-1.0pr.csv mel-sine 767 WARNING
count_lines_with_label ${wavpath_noext}-predicted-1.0pr.csv ambient 124 WARNING

cmd="${srcdir}/${detect_plugin}.py\
      --filename=${wavpath_noext}.wav \
      --parameters='$detect_parameters' \
      --time_units=$time_units \
      --freq_units=$freq_units \
      --time_scale=$time_scale \
      --freq_scale=$freq_scale \
      --audio_tic_rate=$audio_tic_rate \
      --audio_nchannels=$audio_nchannels \
      --audio_read_plugin=$audio_read_plugin \
      --audio_read_plugin_kwargs=$audio_read_plugin_kwargs"
echo $cmd >> ${wavpath_noext}-detect.log 2>&1
eval $cmd >> ${wavpath_noext}-detect.log 2>&1

check_file_exists ${wavpath_noext}-detect.log
check_file_exists ${wavpath_noext}-detected.csv
count_lines_with_label ${wavpath_noext}-detected.csv time 1298 ERROR
count_lines_with_label ${wavpath_noext}-detected.csv frequency 179 ERROR

csvfiles=${wavpath_noext}-detected.csv,${wavpath_noext}-predicted-1.0pr.csv
cmd="${srcdir}/misses $csvfiles"
echo $cmd >> ${wavpath_noext}-misses.log 2>&1
eval $cmd >> ${wavpath_noext}-misses.log 2>&1

check_file_exists ${wavpath_noext}-misses.log
check_file_exists ${wavpath_noext}-missed.csv
count_lines_with_label ${wavpath_noext}-missed.csv other 1569 WARNING

model=train_${ireplicates}r
kinds_touse=annotated,missed
equalize_ratio=1000
max_sounds=10000
cmd="${srcdir}/activations \
      --context=$context \
      --loss=$loss \
      --overlapped_prefix=$overlapped_prefix \
      --shiftby=$shiftby \
      --video_findfile=$video_findfile_plugin \
      --video_bkg_frames=$video_bkg_frames \
      --data_loader_queuesize=$data_loader_queuesize \
      --data_loader_maxprocs=$data_loader_maxprocs \
      --model_architecture=$architecture \
      --model_parameters='$model_parameters' \
      --start_checkpoint=$logdir/$model/ckpt-$nsteps \
      --data_dir=$data_dir \
      --labels_touse=$labels_touse \
      --kinds_touse=$kinds_touse \
      --testing_equalize_ratio=$equalize_ratio \
      --testing_max_sounds=$max_sounds \
      --batch_size=$mini_batch \
      --time_units=$time_units \
      --freq_units=$freq_units \
      --time_scale=$time_scale \
      --freq_scale=$freq_scale \
      --audio_tic_rate=$audio_tic_rate \
      --nchannels=$audio_nchannels \
      --validation_percentage=0.0 \
      --validation_offset_percentage=0.0 \
      --deterministic=$deterministic \
      --save_activations=True \
      --igpu="
echo $cmd >> $data_dir/activations.log 2>&1
eval $cmd >> $data_dir/activations.log 2>&1

check_file_exists $data_dir/activations.log
check_file_exists $data_dir/activations.npz

groundtruth_directory=$data_dir
these_layers=2,3
pca_batch_size=0
cluster_parallelize=1
cluster_parameters='{"ndims":3,"pca-fraction":1.0,"neighbors":10,"distance":0.1}'
cmd="${srcdir}/${cluster_plugin}.py \
     --data_dir=$groundtruth_directory \
     --layers=$these_layers \
     --pca_batch_size=$pca_batch_size \
     --parallelize=$cluster_parallelize \
     --parameters='$cluster_parameters'"
echo $cmd >> $data_dir/cluster.log 2>&1
eval $cmd >> $data_dir/cluster.log 2>&1

check_file_exists $data_dir/cluster.log
check_file_exists $data_dir/cluster.npz

cp $repo_path/data/20161207T102314_ch1-annotated-person1.csv \
   $repo_path/test/scratch/tutorial-sh/groundtruth-data/round2

logdir=$repo_path/test/scratch/tutorial-sh/omit-one
wavfiles=(PS_20130625111709_ch3.wav 20161207T102314_ch1.wav)
mkdir $logdir
ioffsets=$(seq 0 $(( ${#wavfiles[@]} - 1 )) )
for ioffset in $ioffsets ; do
  cmd="${srcdir}/generalize \
       --context=$context \
       --shiftby=$shiftby \
       --optimizer=$optimizer \
       --loss=$loss \
       --overlapped_prefix=$overlapped_prefix \
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
       --save_and_validate_period=$save_and_validate_period \
       --mini_batch=$mini_batch \
       --testing_files='$testing_files' \
       --time_units=$time_units \
       --freq_units=$freq_units \
       --time_scale=$time_scale \
       --freq_scale=$freq_scale \
       --audio_tic_rate=$audio_tic_rate \
       --audio_nchannels=$audio_nchannels \
       --video_frame_rate=$video_frame_rate \
       --video_frame_width=$video_frame_width \
       --video_frame_height=$video_frame_height \
       --video_channels=$video_channels \
       --batch_seed=$batch_seed \
       --weights_seed=$weights_seed \
       --deterministic=$deterministic ''  \
       --igpu=\
       --ioffset=$ioffset \
       --subsets=${wavfiles[ioffset]}"
  echo $cmd >> $logdir/generalize$(( ${ioffset} + 1 )).log 2>&1
  eval $cmd >> $logdir/generalize$(( ${ioffset} + 1 )).log 2>&1
done

for ioffset in $ioffsets ; do
  ioffset1=$(( ${ioffset} + 1 ))
  check_file_exists $logdir/generalize${ioffset1}.log
  check_file_exists $logdir/generalize_${ioffset1}w.log
  check_file_exists $logdir/generalize_${ioffset1}w/ckpt-$nsteps.index
  check_file_exists $logdir/generalize_${ioffset1}w/logits.validation.ckpt-$nsteps.npz
done

cmd="${srcdir}/accuracy \
     --logdir=$logdir \
     --loss=$loss \
     --overlapped_prefix=$overlapped_prefix \
     --error_ratios=$precision_recall_ratios \
     --nprobabilities=$nprobabilities \
     --parallelize=$accuracy_parallelize"
echo $cmd >> $logdir/accuracy.log 2>&1
eval $cmd >> $logdir/accuracy.log 2>&1

check_file_exists $logdir/accuracy.log
check_file_exists $logdir/precision-recall.pdf
check_file_exists $logdir/confusion-matrix.pdf
for ioffset in $ioffsets ; do
  ioffset1=$(( ${ioffset} + 1 ))
  check_file_exists $logdir/generalize_${ioffset1}w/precision-recall.ckpt-$nsteps.pdf
  check_file_exists $logdir/generalize_${ioffset1}w/probability-density.ckpt-$nsteps.pdf
  check_file_exists $logdir/generalize_${ioffset1}w/thresholds.ckpt-$nsteps.csv
  check_file_exists $logdir/generalize_${ioffset1}w/confusion-matrix.ckpt-$nsteps.pdf
done
check_file_exists $logdir/train-validation-loss.pdf
check_file_exists $logdir/P-R-F1-average.pdf
check_file_exists $logdir/P-R-F1-label.pdf
check_file_exists $logdir/P-R-F1-model.pdf
check_file_exists $logdir/PvR.pdf

nfeaturess=(32,32 64,64)
losses=(exclusive overlapped)
precision_recall_ratios=1.0
cp $repo_path/data/PS_20130625111709_ch3-annotated-notsong.csv \
   $repo_path/test/scratch/tutorial-sh/groundtruth-data/round1
cp $repo_path/data/20161207T102314_ch1-annotated-notsong.csv \
   $repo_path/test/scratch/tutorial-sh/groundtruth-data/round2
for loss in ${losses[@]} ; do
    if [ "$loss" == "exclusive" ]; then
        labels_touse=mel-pulse,mel-sine,ambient
    else
        labels_touse=mel-pulse,mel-sine
    fi;
    for nfeatures in ${nfeaturess[@]} ; do
        logdir=$repo_path/test/scratch/tutorial-sh/nfeatures$loss-${nfeatures%%,*}
        kfold=2
        ifolds=$(seq 1 $kfold)
        mkdir $logdir
        for ifold in $ifolds ; do
            cmd="${srcdir}/xvalidate \
                 --context=$context \
                 --shiftby=$shiftby \
                 --optimizer=$optimizer \
                 --loss=$loss \
                 --overlapped_prefix=$overlapped_prefix \
                 --learning_rate=$learning_rate  \
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
                 --save_and_validate_period=$save_and_validate_period \
                 --mini_batch=$mini_batch \
                 --testing_files='$testing_files' \
                 --time_units=$time_units \
                 --freq_units=$freq_units \
                 --time_scale=$time_scale \
                 --freq_scale=$freq_scale \
                 --audio_tic_rate=$audio_tic_rate \
                 --audio_nchannels=$audio_nchannels \
                 --video_frame_rate=$video_frame_rate \
                 --video_frame_width=$video_frame_width \
                 --video_frame_height=$video_frame_height \
                 --video_channels=$video_channels \
                 --batch_seed=$batch_seed \
                 --weights_seed=$weights_seed \
                 --deterministic=$deterministic \
                 --kfold=$kfold \
                 --ifolds=$ifold \
                 --igpu="
            echo $cmd >> $logdir/xvalidate${ifold}.log 2>&1
            eval $cmd >> $logdir/xvalidate${ifold}.log 2>&1
        done
    
        for ifold in $ifolds ; do
            check_file_exists $logdir/xvalidate${ifold}.log
            check_file_exists $logdir/xvalidate_${ifold}k.log
            check_file_exists $logdir/xvalidate_${ifold}k/ckpt-$nsteps.index
            check_file_exists $logdir/xvalidate_${ifold}k/logits.validation.ckpt-$nsteps.npz
        done
    
        cmd="${srcdir}/accuracy \
             --logdir=$logdir \
             --loss=$loss \
             --overlapped_prefix=$overlapped_prefix \
             --error_ratios=$precision_recall_ratios \
             --nprobabilities=$nprobabilities \
             --parallelize=$accuracy_parallelize"
        echo $cmd >> $logdir/accuracy.log 2>&1
        $cmd >> $logdir/accuracy.log 2>&1
    
        check_file_exists $logdir/accuracy.log
        check_file_exists $logdir/precision-recall.pdf
        check_file_exists $logdir/confusion-matrix.pdf
        for ifold in $ifolds ; do
            check_file_exists $logdir/xvalidate_${ifold}k/precision-recall.ckpt-$nsteps.pdf
            check_file_exists $logdir/xvalidate_${ifold}k/probability-density.ckpt-$nsteps.pdf
            check_file_exists $logdir/xvalidate_${ifold}k/thresholds.ckpt-$nsteps.csv
            check_file_exists $logdir/xvalidate_${ifold}k/confusion-matrix.ckpt-$nsteps.pdf
        done
        check_file_exists $logdir/train-validation-loss.pdf
        check_file_exists $logdir/P-R-F1-average.pdf
        check_file_exists $logdir/P-R-F1-label.pdf
        check_file_exists $logdir/P-R-F1-model.pdf
        check_file_exists $logdir/PvR.pdf
    done

    logdirs_prefix=$repo_path/test/scratch/tutorial-sh/nfeatures$loss
    cmd="${srcdir}/compare \
         --logdirs_prefix=$logdirs_prefix \
         --loss=$loss \
         --overlapped_prefix=$overlapped_prefix"
    echo $cmd >> ${logdirs_prefix}-compare.log 2>&1
    eval $cmd >> ${logdirs_prefix}-compare.log 2>&1

    check_file_exists ${logdirs_prefix}-compare.log
    check_file_exists ${logdirs_prefix}-compare-PR-classes.pdf
    check_file_exists ${logdirs_prefix}-compare-confusion-matrices.pdf
    check_file_exists ${logdirs_prefix}-compare-overall-params-speed.pdf
done
loss=exclusive
labels_touse=mel-pulse,mel-sine,ambient

cmd="${srcdir}/mistakes $data_dir"
echo $cmd >> $data_dir/mistakes.log 2>&1
eval $cmd >> $data_dir/mistakes.log 2>&1

check_file_exists $data_dir/mistakes.log
check_file_exists $data_dir/round1/PS_20130625111709_ch3-mistakes.csv

logdir=$repo_path/test/scratch/tutorial-sh/trained-classifier2
kinds_touse=annotated
validation_percentage=20
mkdir $logdir
cmd="${srcdir}/train \
     --context=$context \
     --shiftby=$shiftby \
     --optimizer=$optimizer \
     --loss=$loss \
     --overlapped_prefix=$overlapped_prefix \
     --learning_rate=$learning_rate  \
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
     --save_and_validate_period=$save_and_validate_period \
     --validation_percentage=$validation_percentage \
     --mini_batch=$mini_batch \
     --testing_files='$testing_files' \
     --time_units=$time_units \
     --freq_units=$freq_units \
     --time_scale=$time_scale \
     --freq_scale=$freq_scale \
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
     --igpu="
echo $cmd >> $logdir/train1.log 2>&1
eval $cmd >> $logdir/train1.log 2>&1

check_file_exists $logdir/train1.log
check_file_exists $logdir/train_1r.log
check_file_exists $logdir/train_1r/ckpt-$nsteps.index
check_file_exists $logdir/train_1r/logits.validation.ckpt-$nsteps.npz

cmd="${srcdir}/accuracy \
     --logdir=$logdir \
     --loss=$loss \
     --overlapped_prefix=$overlapped_prefix \
     --error_ratios=$precision_recall_ratios \
     --nprobabilities=$nprobabilities \
     --parallelize=$accuracy_parallelize"
echo $cmd >> $logdir/accuracy.log 2>&1
eval $cmd >> $logdir/accuracy.log 2>&1

check_file_exists $logdir/accuracy.log
check_file_exists $logdir/precision-recall.pdf
check_file_exists $logdir/confusion-matrix.pdf
check_file_exists $logdir/train_1r/precision-recall.ckpt-$nsteps.pdf
check_file_exists $logdir/train_1r/probability-density.ckpt-$nsteps.pdf
check_file_exists $logdir/train_1r/thresholds.ckpt-$nsteps.csv
check_file_exists $logdir/train_1r/confusion-matrix.ckpt-$nsteps.pdf
check_file_exists $logdir/train-validation-loss.pdf
check_file_exists $logdir/P-R-F1-average.pdf
check_file_exists $logdir/P-R-F1-label.pdf
check_file_exists $logdir/P-R-F1-model.pdf
check_file_exists $logdir/PvR.pdf

cmd="${srcdir}/freeze \
      --context=$context \
      --loss=$loss \
      --model_architecture=$architecture \
      --model_parameters='$model_parameters' \
      --start_checkpoint=${logdir}/train_${ireplicates}r/ckpt-$nsteps \
      --output_file=${logdir}/train_${ireplicates}r/frozen-graph.ckpt-${nsteps}.pb \
      --labels_touse=$labels_touse \
      --parallelize=$classify_parallelize \
      --time_units=$time_units \
      --freq_units=$freq_units \
      --time_scale=$time_scale \
      --freq_scale=$freq_scale \
      --audio_tic_rate=$audio_tic_rate \
      --audio_nchannels=$audio_nchannels \
      --video_frame_rate=$video_frame_rate \
      --video_frame_height=$video_frame_height \
      --video_frame_width=$video_frame_width \
      --video_channels=$video_channels \
      --igpu="
echo $cmd >> $logdir/train_${ireplicates}r/freeze.ckpt-${nsteps}.log 2>&1
eval $cmd >> $logdir/train_${ireplicates}r/freeze.ckpt-${nsteps}.log 2>&1

check_file_exists $logdir/train_${ireplicates}r/freeze.ckpt-${nsteps}.log
check_file_exists $logdir/train_${ireplicates}r/frozen-graph.ckpt-${nsteps}.pb

mkdir $repo_path/test/scratch/tutorial-sh/groundtruth-data/dense
cp $repo_path/data/20190122T093303a-7.wav \
   $repo_path/test/scratch/tutorial-sh/groundtruth-data/dense

wavpath_noext=$repo_path/test/scratch/tutorial-sh/groundtruth-data/dense/20190122T093303a-7
cmd="${srcdir}/classify \
      --context=$context \
      --loss=$loss \
      --shiftby=$shiftby \
      --audio_read_plugin=$audio_read_plugin \
      --audio_read_plugin_kwargs=$audio_read_plugin_kwargs \
      --video_read_plugin=$video_read_plugin \
      --video_read_plugin_kwargs=$video_read_plugin_kwargs \
      --video_findfile_plugin=$video_findfile_plugin \
      --video_bkg_frames=$video_bkg_frames \
      --model=$logdir/train_${ireplicates}r/frozen-graph.ckpt-${nsteps}.pb \
      --model_labels=$logdir/train_${ireplicates}r/labels.txt \
      --wav=${wavpath_noext}.wav \
      --parallelize=$classify_parallelize \
      --time_scale=$time_scale \
      --audio_tic_rate=$audio_tic_rate \
      --audio_nchannels=$audio_nchannels \
      --video_frame_rate=$video_frame_rate \
      --video_frame_height=$video_frame_height \
      --video_frame_width=$video_frame_width \
      --video_channels=$video_channels \
      --deterministic=$deterministic \
      --labels= \
      --prevalences= \
      --igpu="
echo $cmd >> ${wavpath_noext}-classify.log 2>&1
eval $cmd >> ${wavpath_noext}-classify.log 2>&1

check_file_exists ${wavpath_noext}-classify.log

for label in $(echo $labels_touse | sed "s/,/ /g") ; do
  check_file_exists ${wavpath_noext}-${label}.wav
done

cmd="${srcdir}/ethogram \
      $logdir train_${ireplicates}r thresholds.ckpt-${nsteps}.csv \
      ${wavpath_noext}.wav $audio_tic_rate"
echo $cmd >> ${wavpath_noext}-ethogram.log 2>&1
eval $cmd >> ${wavpath_noext}-ethogram.log 2>&1

check_file_exists ${wavpath_noext}-ethogram.log
for pr in $(echo $precision_recall_ratios | sed "s/,/ /g") ; do
  check_file_exists ${wavpath_noext}-predicted-${pr}pr.csv
done

wav_file_noext=20190122T093303a-7
cp $repo_path/data/${wav_file_noext}-annotated-person2.csv \
   $repo_path/test/scratch/tutorial-sh/groundtruth-data/dense
cp $repo_path/data/${wav_file_noext}-annotated-person3.csv \
   $repo_path/test/scratch/tutorial-sh/groundtruth-data/dense

portion=union
convolve=0.0
measure=both
congruence_dir=$data_dir/congruence-11112233T445566
mkdir $congruence_dir
cmd="${srcdir}/congruence \
     --basepath=$data_dir \
     --topath=$congruence_dir \
     --wavfiles=${wav_file_noext}.wav \
     --portion=$portion \
     --convolve=$convolve \
     --measure=$measure \
     --nprobabilities=$nprobabilities \
     --audio_tic_rate=$audio_tic_rate \
     --parallelize=$congruence_parallelize"
echo $cmd >> $congruence_dir/congruence.log 2>&1
eval $cmd >> $congruence_dir/congruence.log 2>&1

check_file_exists $congruence_dir/congruence.log
check_file_exists $congruence_dir/dense/$wav_file_noext-disjoint-everyone.csv
kinds=(tic label)
persons=(person2 person3)
IFS=', ' read -r -a prs <<< "$precision_recall_ratios"
IFS=', ' read -r -a labels <<< "$labels_touse"
for kind in ${kinds[@]} ; do
  for label in ${labels[@]} ; do
    check_file_exists $congruence_dir/congruence.${kind}.${label}.csv
    count_lines $congruence_dir/congruence.${kind}.${label}.csv $(( $nprobabilities + 2 ))
    check_file_exists $congruence_dir/congruence.${kind}.${label}.pdf
  done
  for pr in ${prs[@]} ; do
    for label in ${labels[@]} ; do
      check_file_exists $congruence_dir/congruence.${kind}.${label}.${pr}pr-venn.pdf
      check_file_exists $congruence_dir/congruence.${kind}.${label}.${pr}pr.pdf
    done
    check_file_exists $congruence_dir/dense/$wav_file_noext-disjoint-${kind}-not${pr}pr.csv
    check_file_exists $congruence_dir/dense/$wav_file_noext-disjoint-${kind}-only${pr}pr.csv
  done
  for person in ${persons[@]} ; do
    check_file_exists $congruence_dir/dense/$wav_file_noext-disjoint-${kind}-not${person}.csv
    check_file_exists $congruence_dir/dense/$wav_file_noext-disjoint-${kind}-only${person}.csv
  done
done

logdir=${repo_path}/test/scratch/tutorial-sh/nfeaturesexclusive-64

mkdir ${logdir}/xvalidate_1k_2k
cmd="${srcdir}/ensemble \
      --start_checkpoints=${logdir}/xvalidate_1k/ckpt-${nsteps},${logdir}/xvalidate_2k/ckpt-${nsteps} \
      --output_file=${logdir}/xvalidate_1k_2k/frozen-graph.ckpt-${nsteps}_${nsteps}.pb \
      --labels_touse=mel-pulse,mel-sine,ambient \
      --context=$context \
      --model_architecture=$architecture \
      --model_parameters='$model_parameters' \
      --parallelize=$classify_parallelize \
      --time_units=$time_units \
      --freq_units=$freq_units \
      --time_scale=$time_scale \
      --freq_scale=$freq_scale \
      --audio_tic_rate=$audio_tic_rate \
      --nchannels=$audio_nchannels"
echo $cmd >> ${logdir}/xvalidate_1k_2k/ensemble.log 2>&1
eval $cmd >> ${logdir}/xvalidate_1k_2k/ensemble.log 2>&1

check_file_exists ${logdir}/xvalidate_1k_2k/ensemble.log
check_file_exists ${logdir}/xvalidate_1k_2k/frozen-graph.ckpt-${nsteps}_${nsteps}.pb/saved_model.pb 

mkdir -p $repo_path/test/scratch/tutorial-sh/groundtruth-data/dense-ensemble
cp $repo_path/data/20190122T132554a-14.wav \
   $repo_path/test/scratch/tutorial-sh/groundtruth-data/dense-ensemble

wavpath_noext=$repo_path/test/scratch/tutorial-sh/groundtruth-data/dense-ensemble/20190122T132554a-14
cmd="${srcdir}/classify \
      --context=$context \
      --loss=$loss \
      --shiftby=$shiftby \
      --audio_read_plugin=$audio_read_plugin \
      --audio_read_plugin_kwargs=$audio_read_plugin_kwargs \
      --video_read_plugin=$video_read_plugin \
      --video_read_plugin_kwargs=$video_read_plugin_kwargs \
      --video_findfile_plugin=$video_findfile_plugin \
      --video_bkg_frames=$video_bkg_frames \
      --model=${logdir}/xvalidate_1k_2k/frozen-graph.ckpt-${nsteps}_${nsteps}.pb \
      --model_labels=${logdir}/xvalidate_1k_2k/labels.txt \
      --wav=${wavpath_noext}.wav \
      --parallelize=$classify_parallelize \
      --time_scale=$time_scale \
      --audio_tic_rate=$audio_tic_rate \
      --audio_nchannels=$audio_nchannels \
      --video_frame_rate=$video_frame_rate \
      --video_frame_height=$video_frame_height \
      --video_frame_width=$video_frame_width \
      --video_channels=$video_channels \
      --deterministic=$deterministic \
      --labels= \
      --prevalences= \
      --igpu="
echo $cmd >> ${wavpath_noext}-classify.log 2>&1
eval $cmd >> ${wavpath_noext}-classify.log 2>&1

check_file_exists ${wavpath_noext}-classify.log

for label in $(echo $labels_touse | sed "s/,/ /g") ; do
  check_file_exists ${wavpath_noext}-${label}.wav
done

cmd="${srcdir}/ethogram \
      $logdir xvalidate_1k thresholds.ckpt-${nsteps}.csv \
      ${wavpath_noext}.wav $audio_tic_rate"
echo $cmd >> ${wavpath_noext}-ethogram.log 2>&1
eval $cmd >> ${wavpath_noext}-ethogram.log 2>&1

check_file_exists ${wavpath_noext}-ethogram.log
for pr in $(echo $precision_recall_ratios | sed "s/,/ /g") ; do
  check_file_exists ${wavpath_noext}-predicted-${pr}pr.csv
done
count_lines_with_label ${wavpath_noext}-predicted-1.0pr.csv mel-pulse 56 WARNING
count_lines_with_label ${wavpath_noext}-predicted-1.0pr.csv mel-sine 140 WARNING
count_lines_with_label ${wavpath_noext}-predicted-1.0pr.csv ambient 70 WARNING

wav_file_noext=20190122T132554a-14
cp $repo_path/data/${wav_file_noext}-annotated-person2.csv \
   $repo_path/test/scratch/tutorial-sh/groundtruth-data/dense-ensemble
cp $repo_path/data/${wav_file_noext}-annotated-person3.csv \
   $repo_path/test/scratch/tutorial-sh/groundtruth-data/dense-ensemble

congruence_dir=$data_dir/congruence-99998877T665544
mkdir $congruence_dir
cmd="${srcdir}/congruence \
     --basepath=$data_dir \
     --topath=$congruence_dir \
     --wavfiles=${wav_file_noext}.wav \
     --portion=$portion \
     --convolve=$convolve \
     --measure=$measure \
     --nprobabilities=$nprobabilities \
     --audio_tic_rate=$audio_tic_rate \
     --parallelize=$congruence_parallelize"
echo $cmd >> $congruence_dir/congruence.log 2>&1
eval $cmd >> $congruence_dir/congruence.log 2>&1

check_file_exists $congruence_dir/congruence.log
check_file_exists $congruence_dir/dense-ensemble/$wav_file_noext-disjoint-everyone.csv
kinds=(tic label)
persons=(person2 person3)
IFS=', ' read -r -a prs <<< "$precision_recall_ratios"
IFS=', ' read -r -a labels <<< "$labels_touse"
for kind in ${kinds[@]} ; do
  for label in ${labels[@]} ; do
    check_file_exists $congruence_dir/congruence.${kind}.${label}.csv
    count_lines $congruence_dir/congruence.${kind}.${label}.csv $(( $nprobabilities + 2 ))
    check_file_exists $congruence_dir/congruence.${kind}.${label}.pdf
  done
  for pr in ${prs[@]} ; do
    for label in ${labels[@]} ; do
      check_file_exists $congruence_dir/congruence.${kind}.${label}.${pr}pr-venn.pdf
      check_file_exists $congruence_dir/congruence.${kind}.${label}.${pr}pr.pdf
    done
    check_file_exists $congruence_dir/dense-ensemble/$wav_file_noext-disjoint-${kind}-not${pr}pr.csv
    check_file_exists $congruence_dir/dense-ensemble/$wav_file_noext-disjoint-${kind}-only${pr}pr.csv
  done
  for person in ${persons[@]} ; do
    check_file_exists $congruence_dir/dense-ensemble/$wav_file_noext-disjoint-${kind}-not${person}.csv
    check_file_exists $congruence_dir/dense-ensemble/$wav_file_noext-disjoint-${kind}-only${person}.csv
  done
done

wavpath_noext=$repo_path/test/scratch/tutorial-sh/groundtruth-data/round1/PS_20130625111709_ch3
cmd="${srcdir}/classify \
      --context=$context \
      --loss=$loss \
      --shiftby=$shiftby \
      --audio_read_plugin=$audio_read_plugin \
      --audio_read_plugin_kwargs=$audio_read_plugin_kwargs \
      --video_read_plugin=$video_read_plugin \
      --video_read_plugin_kwargs=$video_read_plugin_kwargs \
      --video_findfile_plugin=$video_findfile_plugin \
      --video_bkg_frames=$video_bkg_frames \
      --model=${logdir}/xvalidate_1k_2k/frozen-graph.ckpt-${nsteps}_${nsteps}.pb \
      --model_labels=${logdir}/xvalidate_1k_2k/labels.txt \
      --wav=${wavpath_noext}.wav \
      --parallelize=$classify_parallelize \
      --time_scale=$time_scale \
      --audio_tic_rate=$audio_tic_rate \
      --audio_nchannels=$audio_nchannels \
      --video_frame_rate=$video_frame_rate \
      --video_frame_height=$video_frame_height \
      --video_frame_width=$video_frame_width \
      --video_channels=$video_channels \
      --deterministic=$deterministic \
      --labels= \
      --prevalences= \
      --igpu="
echo $cmd >> ${wavpath_noext}-classify.log 2>&1
eval $cmd >> ${wavpath_noext}-classify.log 2>&1

check_file_exists ${wavpath_noext}-classify.log

for label in $(echo $labels_touse | sed "s/,/ /g") ; do
  check_file_exists ${wavpath_noext}-${label}.wav
done

thresholds_dense_file=$(basename $(ls ${logdir}/xvalidate_1k/thresholds-dense-*))
mv ${logdir}/xvalidate_1k/${thresholds_dense_file} ${logdir}/xvalidate_1k_2k

cmd="${srcdir}/ethogram \
      $logdir xvalidate_1k_2k ${thresholds_dense_file} \
      ${wavpath_noext}.wav $audio_tic_rate"
echo $cmd >> ${wavpath_noext}-ethogram.log 2>&1
eval $cmd >> ${wavpath_noext}-ethogram.log 2>&1

check_file_exists ${wavpath_noext}-ethogram.log
for pr in $(echo $precision_recall_ratios | sed "s/,/ /g") ; do
  check_file_exists ${wavpath_noext}-predicted-${pr}pr.csv
done
count_lines_with_label ${wavpath_noext}-predicted-1.0pr.csv mel-pulse 594 WARNING
