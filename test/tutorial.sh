#!/bin/bash

# recapitulate the tutorial via the shell interface

#${SONGEXPLORER_BIN/-B/-B /tmp:/opt/songexplorer/test/scratch -B} bash -c "test/tutorial.sh"

check_file_exists() {
  if [[ ! -e $1 ]] ; then
    echo ERROR: $1 is missing
    return 1
  fi
  return 0; }

count_lines_with_label() {
  check_file_exists $1 || return
  local count=$(grep $2 $1 | wc -l)
  (( "$count" == "$3" )) && return
  echo $4: $1 has $count $2 when it should have $3
  if [ "$4" == "WARNING" ]; then echo $4: it is normal for this to be close but not exact; fi; }

count_lines() {
  check_file_exists $1 || return
  local count=$(cat $1 | wc -l)
  (( "$count" == "$2" )) && return
  echo ERROR: $1 has $count lines when it should have $2; }

repo_path=$(dirname $(dirname $(which detect.sh)))

mkdir -p $repo_path/test/scratch/tutorial-sh
cp $repo_path/configuration.pysh $repo_path/test/scratch/tutorial-sh

source $repo_path/test/scratch/tutorial-sh/configuration.pysh

mkdir -p $repo_path/test/scratch/tutorial-sh/groundtruth-data/round1
cp $repo_path/data/PS_20130625111709_ch3.wav \
   $repo_path/test/scratch/tutorial-sh/groundtruth-data/round1

wavpath_noext=$repo_path/test/scratch/tutorial-sh/groundtruth-data/round1/PS_20130625111709_ch3
time_sigma_signal=9
time_sigma_noise=4
time_smooth_ms=6.4
frequency_n_ms=25.6
frequency_nw=4
frequency_p_signal=0.1
frequency_p_noise=1.0
frequency_smooth_ms=25.6
detect.sh \
      ${wavpath_noext}.wav \
      $time_sigma_signal $time_sigma_noise $time_smooth_ms \
      $frequency_n_ms $frequency_nw $frequency_p_signal $frequency_p_noise $frequency_smooth_ms \
      $audio_tic_rate $audio_nchannels \
      &> ${wavpath_noext}-detect.log

check_file_exists ${wavpath_noext}-detect.log
check_file_exists ${wavpath_noext}-detected.csv
count_lines_with_label ${wavpath_noext}-detected.csv time 536 ERROR
count_lines_with_label ${wavpath_noext}-detected.csv frequency 45 ERROR
count_lines_with_label ${wavpath_noext}-detected.csv neither 1635 ERROR

context_ms=204.8
shiftby_ms=0.0
representation=mel-cepstrum
window_ms=6.4
stride_ms=1.6
mel=7
dct=7
optimizer=Adam
learning_rate=0.0002
architecture=convolutional
model_parameters='{"dropout": "0.5", "kernel_sizes": "5,3", "nlayers": "2", "nfeatures": "64,64", "dilate_after_layer": "65535", "stride_after_layer": "65535", "connection_type": "plain"}'
logdir=$repo_path/test/scratch/tutorial-sh/untrained-classifier
data_dir=$repo_path/test/scratch/tutorial-sh/groundtruth-data
labels_touse=time,frequency
kinds_touse=detected
nsteps=0
restore_from=''
save_and_test_period=0
validation_percentage=0
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
      $logdir $data_dir $labels_touse $kinds_touse \
      $nsteps "$restore_from" $save_and_test_period $validation_percentage \
      $mini_batch "$testing_files" \
      $audio_tic_rate $audio_nchannels \
      $batch_seed $weights_seed $ireplicates \
      &> $logdir/train1.log

check_file_exists $logdir/train1.log
check_file_exists $logdir/train_1r.log
check_file_exists $logdir/train_1r/ckpt-$nsteps.index

check_point=$nsteps
equalize_ratio=1000
max_sounds=10000
activations.sh \
      $context_ms $shiftby_ms $representation $window_ms $stride_ms $mel $dct \
      $architecture "$model_parameters" \
      $logdir train_${ireplicates}r $check_point \
      $data_dir $labels_touse $kinds_touse \
      $equalize_ratio $max_sounds $mini_batch \
      $audio_tic_rate $audio_nchannels \
      &> $data_dir/activations.log

check_file_exists $data_dir/activations.log
check_file_exists $data_dir/activations-sounds.log
check_file_exists $data_dir/activations.npz

groundtruth_directory=$data_dir
these_layers=0
pca_fraction_variance_to_retain=0.99
pca_batch_size=0
cluster_algorithm=tsne
cluster_ndims=2
cluster_args=(30 12)
cluster.sh \
      $groundtruth_directory $these_layers \
      $pca_fraction_variance_to_retain $pca_batch_size \
      $cluster_algorithm $cluster_ndims $cluster_parallelize ${cluster_args[@]} \
      &> $data_dir/cluster.log

check_file_exists $data_dir/cluster.log
check_file_exists $data_dir/cluster.npz
check_file_exists $data_dir/cluster-pca.pdf

cp $repo_path/data/PS_20130625111709_ch3-annotated-person1.csv \
   $repo_path/test/scratch/tutorial-sh/groundtruth-data/round1

logdir=$repo_path/test/scratch/tutorial-sh/trained-classifier1
labels_touse=mel-pulse,mel-sine,ambient
kinds_touse=annotated
nsteps=100
save_and_test_period=10
validation_percentage=40
mkdir $logdir
train.sh \
      $context_ms $shiftby_ms $representation $window_ms $stride_ms $mel $dct \
      $optimizer $learning_rate  \
      $architecture "$model_parameters" \
      $logdir $data_dir $labels_touse $kinds_touse \
      $nsteps "$restore_from" $save_and_test_period $validation_percentage \
      $mini_batch "$testing_files" \
      $audio_tic_rate $audio_nchannels \
      $batch_seed $weights_seed $ireplicates \
      &> $logdir/train1.log

check_file_exists $logdir/train1.log
check_file_exists $logdir/train_1r.log
check_file_exists $logdir/train_1r/ckpt-$nsteps.index
check_file_exists $logdir/train_1r/logits.validation.ckpt-$nsteps.npz

precision_recall_ratios=0.5,1.0,2.0
accuracy.sh $logdir $precision_recall_ratios \
      $nprobabilities $accuracy_parallelize \
      &> $logdir/accuracy.log

check_file_exists $logdir/accuracy.log
check_file_exists $logdir/accuracy.pdf
check_file_exists $logdir/train_1r/precision-recall.ckpt-$nsteps.pdf
check_file_exists $logdir/train_1r/probability-density.ckpt-$nsteps.pdf
check_file_exists $logdir/train_1r/thresholds.ckpt-$nsteps.csv
check_file_exists $logdir/train-loss.pdf
check_file_exists $logdir/validation-F1.pdf
for label in $(echo $labels_touse | sed "s/,/ /g") ; do
  check_file_exists $logdir/validation-PvR-$label.pdf
done

check_point=$nsteps
freeze.sh \
      $context_ms $representation $window_ms $stride_ms $mel $dct \
      $architecture "$model_parameters" \
      $logdir train_${ireplicates}r $check_point $nwindows \
      $audio_tic_rate $audio_nchannels \
      &> $logdir/train_${ireplicates}r/freeze.ckpt-$check_point.log

check_file_exists $logdir/train_${ireplicates}r/freeze.ckpt-$check_point.log
check_file_exists $logdir/train_${ireplicates}r/frozen-graph.ckpt-$check_point.pb

mkdir $repo_path/test/scratch/tutorial-sh/groundtruth-data/round2
cp $repo_path/data/20161207T102314_ch1.wav \
   $repo_path/test/scratch/tutorial-sh/groundtruth-data/round2

wavpath_noext=$repo_path/test/scratch/tutorial-sh/groundtruth-data/round2/20161207T102314_ch1
classify1.sh \
      $context_ms '' $representation $stride_ms \
      $logdir train_${ireplicates}r $check_point \
      ${wavpath_noext}.wav \
      $audio_tic_rate $nwindows &> ${wavpath_noext}-classify1.log

check_file_exists ${wavpath_noext}.tf
check_file_exists ${wavpath_noext}-classify1.log

classify2.sh \
      $context_ms $shiftby_ms $representation $stride_ms \
      $logdir train_${ireplicates}r $check_point \
      ${wavpath_noext}.wav $audio_tic_rate \
      &> ${wavpath_noext}-classify2.log

check_file_exists ${wavpath_noext}-classify2.log
for label in $(echo $labels_touse | sed "s/,/ /g") ; do
  check_file_exists ${wavpath_noext}-${label}.wav
done

ethogram.sh \
      $logdir train_${ireplicates}r thresholds.ckpt-${check_point}.csv \
      $wavpath_noext $audio_tic_rate \
      &> ${wavpath_noext}-ethogram.log

check_file_exists ${wavpath_noext}-ethogram.log
for pr in $(echo $precision_recall_ratios | sed "s/,/ /g") ; do
  check_file_exists ${wavpath_noext}-predicted-${pr}pr.csv
done
count_lines_with_label ${wavpath_noext}-predicted-1.0pr.csv mel-pulse 416 WARNING
count_lines_with_label ${wavpath_noext}-predicted-1.0pr.csv mel-sine 439 WARNING
count_lines_with_label ${wavpath_noext}-predicted-1.0pr.csv ambient 212 WARNING

detect.sh \
      ${wavpath_noext}.wav \
      $time_sigma_signal $time_sigma_noise $time_smooth_ms \
      $frequency_n_ms $frequency_nw $frequency_p_signal $frequency_p_noise $frequency_smooth_ms \
      $audio_tic_rate $audio_nchannels \
      &> ${wavpath_noext}-detect.log

check_file_exists ${wavpath_noext}-detect.log
check_file_exists ${wavpath_noext}-detected.csv
count_lines_with_label ${wavpath_noext}-detected.csv time 1298 ERROR
count_lines_with_label ${wavpath_noext}-detected.csv frequency 179 ERROR

csvfiles=${wavpath_noext}-detected.csv,${wavpath_noext}-predicted-1.0pr.csv
misses.sh $csvfiles &> ${wavpath_noext}-misses.log

check_file_exists ${wavpath_noext}-misses.log
check_file_exists ${wavpath_noext}-missed.csv
count_lines_with_label ${wavpath_noext}-missed.csv other 1607 WARNING

mkdir $data_dir/round1/cluster
mv $data_dir/{activations,cluster}* $data_dir/round1/cluster

model=train_${ireplicates}r
kinds_touse=annotated,missed
equalize_ratio=1000
max_sounds=10000
activations.sh \
      $context_ms $shiftby_ms $representation $window_ms $stride_ms $mel $dct \
      $architecture "$model_parameters" \
      $logdir $model $check_point \
      $data_dir $labels_touse $kinds_touse \
      $equalize_ratio $max_sounds $mini_batch \
      $audio_tic_rate $audio_nchannels \
      &> $data_dir/activations.log

check_file_exists $data_dir/activations.log
check_file_exists $data_dir/activations-sounds.log
check_file_exists $data_dir/activations.npz

groundtruth_directory=$data_dir
these_layers=2,3
pca_fraction_variance_to_retain=1.0
pca_batch_size=0
cluster_algorithm=umap
cluster_ndims=3
cluster_parallelize=1
cluster_args=(10 0.1)
cluster.sh \
      $groundtruth_directory $these_layers \
      $pca_fraction_variance_to_retain $pca_batch_size \
      $cluster_algorithm $cluster_ndims $cluster_parallelize ${cluster_args[@]} \
      &> $data_dir/cluster.log

check_file_exists $data_dir/cluster.log
check_file_exists $data_dir/cluster.npz

cp $repo_path/data/20161207T102314_ch1-annotated-person1.csv \
   $repo_path/test/scratch/tutorial-sh/groundtruth-data/round2

logdir=$repo_path/test/scratch/tutorial-sh/omit-one
wavfiles=(PS_20130625111709_ch3.wav 20161207T102314_ch1.wav)
mkdir $logdir
ioffsets=$(seq 0 $(dc -e "${#wavfiles[@]} 1 - p"))
for ioffset in $ioffsets ; do
  generalize.sh \
        $context_ms $shiftby_ms $representation $window_ms $stride_ms $mel $dct \
        $optimizer $learning_rate \
        $architecture "$model_parameters" \
        $logdir $data_dir $labels_touse $kinds_touse \
        $nsteps "$restore_from" $save_and_test_period $mini_batch \
        "$testing_files" $audio_tic_rate $audio_nchannels \
        $batch_seed $weights_seed \
        $ioffset ${wavfiles[ioffset]} \
        &> $logdir/generalize$(dc -e "${ioffset} 1 + p").log
done

for ioffset in $ioffsets ; do
  ioffset1=$(dc -e "${ioffset} 1 + p")
  check_file_exists $logdir/generalize${ioffset1}.log
  check_file_exists $logdir/generalize_${ioffset1}w.log
  check_file_exists $logdir/generalize_${ioffset1}w/ckpt-$nsteps.index
  check_file_exists $logdir/generalize_${ioffset1}w/logits.validation.ckpt-$nsteps.npz
done

accuracy.sh $logdir $precision_recall_ratios \
      $nprobabilities $accuracy_parallelize \
      &> $logdir/accuracy.log

check_file_exists $logdir/accuracy.log
check_file_exists $logdir/accuracy.pdf
check_file_exists $logdir/confusion-matrices.pdf
for ioffset in $ioffsets ; do
  ioffset1=$(dc -e "${ioffset} 1 + p")
  check_file_exists $logdir/generalize_${ioffset1}w/precision-recall.ckpt-$nsteps.pdf
  check_file_exists $logdir/generalize_${ioffset1}w/probability-density.ckpt-$nsteps.pdf
  check_file_exists $logdir/generalize_${ioffset1}w/thresholds.ckpt-$nsteps.csv
done
check_file_exists $logdir/train-loss.pdf
check_file_exists $logdir/validation-F1.pdf
for label in $(echo $labels_touse | sed "s/,/ /g") ; do
  check_file_exists $logdir/validation-PvR-$label.pdf
done

nfeaturess=(32,32 64,64)
for nfeatures in ${nfeaturess[@]} ; do
  logdir=$repo_path/test/scratch/tutorial-sh/nfeatures-${nfeatures%%,*}
  kfold=2
  ifolds=$(seq 1 $kfold)
  mkdir $logdir
  for ifold in $ifolds ; do
    xvalidate.sh \
          $context_ms $shiftby_ms $representation $window_ms $stride_ms $mel $dct \
          $optimizer $learning_rate  \
          $architecture "$model_parameters" \
          $logdir $data_dir $labels_touse $kinds_touse \
          $nsteps "$restore_from" $save_and_test_period $mini_batch \
          "$testing_files" $audio_tic_rate $audio_nchannels \
          $batch_seed $weights_seed \
          $kfold $ifold \
          &> $logdir/xvalidate${ifold}.log
  done

  for ifold in $ifolds ; do
    check_file_exists $logdir/xvalidate${ifold}.log
    check_file_exists $logdir/xvalidate_${ifold}k.log
    check_file_exists $logdir/xvalidate_${ifold}k/ckpt-$nsteps.index
    check_file_exists $logdir/xvalidate_${ifold}k/logits.validation.ckpt-$nsteps.npz
  done

  accuracy.sh $logdir $precision_recall_ratios \
        $nprobabilities $accuracy_parallelize \
        &> $logdir/accuracy.log

  check_file_exists $logdir/accuracy.log
  check_file_exists $logdir/accuracy.pdf
  check_file_exists $logdir/confusion-matrices.pdf
  for ifold in $ifolds ; do
    check_file_exists $logdir/xvalidate_${ifold}k/precision-recall.ckpt-$nsteps.pdf
    check_file_exists $logdir/xvalidate_${ifold}k/probability-density.ckpt-$nsteps.pdf
    check_file_exists $logdir/xvalidate_${ifold}k/thresholds.ckpt-$nsteps.csv
  done
  check_file_exists $logdir/train-loss.pdf
  check_file_exists $logdir/validation-F1.pdf
  for label in $(echo $labels_touse | sed "s/,/ /g") ; do
    check_file_exists $logdir/validation-PvR-$label.pdf
  done
done

logdirs_prefix=$repo_path/test/scratch/tutorial-sh/nfeatures
compare.sh $logdirs_prefix &> ${logdirs_prefix}-compare.log

check_file_exists ${logdirs_prefix}-compare.log
check_file_exists ${logdirs_prefix}-compare-precision-recall.pdf
check_file_exists ${logdirs_prefix}-compare-confusion-matrices.pdf
check_file_exists ${logdirs_prefix}-compare-overall-params-speed.pdf

mistakes.sh $data_dir &> $data_dir/mistakes.log

check_file_exists $data_dir/mistakes.log
check_file_exists $data_dir/round1/PS_20130625111709_ch3-mistakes.csv

logdir=$repo_path/test/scratch/tutorial-sh/trained-classifier2
kinds_touse=annotated
nsteps=100
validation_percentage=20
mkdir $logdir
train.sh \
      $context_ms $shiftby_ms $representation $window_ms $stride_ms $mel $dct \
      $optimizer $learning_rate  \
      $architecture "$model_parameters" \
      $logdir $data_dir $labels_touse $kinds_touse \
      $nsteps "$restore_from" $save_and_test_period $validation_percentage \
      $mini_batch "$testing_files" \
      $audio_tic_rate $audio_nchannels \
      $batch_seed $weights_seed $ireplicates \
      &> $logdir/train1.log

check_file_exists $logdir/train1.log
check_file_exists $logdir/train_1r.log
check_file_exists $logdir/train_1r/ckpt-$nsteps.index
check_file_exists $logdir/train_1r/logits.validation.ckpt-$nsteps.npz

precision_recall_ratios=1.0
accuracy.sh $logdir $precision_recall_ratios \
      $nprobabilities $accuracy_parallelize \
      &> $logdir/accuracy.log

check_file_exists $logdir/accuracy.log
check_file_exists $logdir/accuracy.pdf
check_file_exists $logdir/train_1r/precision-recall.ckpt-$nsteps.pdf
check_file_exists $logdir/train_1r/probability-density.ckpt-$nsteps.pdf
check_file_exists $logdir/train_1r/thresholds.ckpt-$nsteps.csv
check_file_exists $logdir/train-loss.pdf
check_file_exists $logdir/validation-F1.pdf
for label in $(echo $labels_touse | sed "s/,/ /g") ; do
  check_file_exists $logdir/validation-PvR-$label.pdf
done

freeze.sh \
      $context_ms $representation $window_ms $stride_ms $mel $dct \
      $architecture "$model_parameters" \
      $logdir train_${ireplicates}r $check_point $nwindows \
      $audio_tic_rate $audio_nchannels \
      &> $logdir/train_${ireplicates}r/freeze.ckpt-$check_point.log

check_file_exists $logdir/train_${ireplicates}r/freeze.ckpt-$check_point.log
check_file_exists $logdir/train_${ireplicates}r/frozen-graph.ckpt-$check_point.pb

mkdir $repo_path/test/scratch/tutorial-sh/groundtruth-data/congruence
cp $repo_path/data/20190122T093303a-7.wav \
   $repo_path/test/scratch/tutorial-sh/groundtruth-data/congruence

wavpath_noext=$repo_path/test/scratch/tutorial-sh/groundtruth-data/congruence/20190122T093303a-7
classify1.sh \
      $context_ms '' $representation $stride_ms \
      $logdir train_${ireplicates}r $check_point ${wavpath_noext}.wav \
      $audio_tic_rate $nwindows \
      &> ${wavpath_noext}-classify1.log

check_file_exists ${wavpath_noext}-classify1.log
check_file_exists ${wavpath_noext}.tf

classify2.sh \
      $context_ms $shiftby_ms $representation $stride_ms \
      $logdir train_${ireplicates}r $check_point \
      ${wavpath_noext}.wav $audio_tic_rate '' \
      &> ${wavpath_noext}-classify2.log

check_file_exists ${wavpath_noext}-classify2.log
for label in $(echo $labels_touse | sed "s/,/ /g") ; do
  check_file_exists ${wavpath_noext}-${label}.wav
done

ethogram.sh \
      $logdir train_${ireplicates}r thresholds.ckpt-${check_point}.csv \
      ${wavpath_noext}.wav $audio_tic_rate \
      &> ${wavpath_noext}-ethogram.log

check_file_exists ${wavpath_noext}-ethogram.log
for pr in $(echo $precision_recall_ratios | sed "s/,/ /g") ; do
  check_file_exists ${wavpath_noext}-predicted-${pr}pr.csv
done

cp $repo_path/data/20190122T093303a-7-annotated-person2.csv \
   $repo_path/test/scratch/tutorial-sh/groundtruth-data/congruence
cp $repo_path/data/20190122T093303a-7-annotated-person3.csv \
   $repo_path/test/scratch/tutorial-sh/groundtruth-data/congruence

wav_file_noext=20190122T093303a-7
portion=union
convolve_ms=0.0
congruence.sh \
      $data_dir ${wav_file_noext}.wav $portion $convolve_ms $nprobabilities $audio_tic_rate $congruence_parallelize \
      &> $data_dir/congruence.log

check_file_exists $data_dir/congruence.log
check_file_exists $data_dir/congruence/$wav_file_noext-disjoint-everyone.csv
kinds=(tic label)
persons=(person2 person3)
IFS=', ' read -r -a prs <<< "$precision_recall_ratios"
IFS=', ' read -r -a labels <<< "$labels_touse"
for kind in ${kinds[@]} ; do
  for label in ${labels[@]} ; do
    check_file_exists $data_dir/congruence.${kind}.${label}.csv
    count_lines $data_dir/congruence.${kind}.${label}.csv $(( $nprobabilities + 2 ))
    check_file_exists $data_dir/congruence.${kind}.${label}.pdf
  done
  for pr in ${prs[@]} ; do
    for label in ${labels[@]} ; do
      check_file_exists $data_dir/congruence.${kind}.${label}.${pr}pr-venn.pdf
      check_file_exists $data_dir/congruence.${kind}.${label}.${pr}pr.pdf
    done
    check_file_exists $data_dir/congruence/$wav_file_noext-disjoint-${kind}-not${pr}pr.csv
    check_file_exists $data_dir/congruence/$wav_file_noext-disjoint-${kind}-only${pr}pr.csv
  done
  for person in ${persons[@]} ; do
    check_file_exists $data_dir/congruence/$wav_file_noext-disjoint-${kind}-not${person}.csv
    check_file_exists $data_dir/congruence/$wav_file_noext-disjoint-${kind}-only${person}.csv
  done
done
