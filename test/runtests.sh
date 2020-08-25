#!/bin/bash

# run all of the tests

# singularity exec -B /tmp:/opt/deepsong/test/scratch <--nv> <deepsong.sif> test/runtests.sh

repo_path=$(dirname $(dirname $(which detect.sh)))

$repo_path/test/tutorial.sh
$repo_path/test/tutorial.py

if (( $(diff <(tree $repo_path/test/scratch/sh | grep -v tfevents) \
             <(tree $repo_path/test/scratch/py | grep -v tfevents) | wc -l) > 4 )) ; then
  echo ERROR
fi
files=(groundtruth-data/round1/PS_20130625111709_ch3-detected.csv
       trained-classifier1/train_1r/thresholds.ckpt-10.csv
       trained-classifier1/train_1r/thresholds.ckpt-100.csv
       groundtruth-data/round2/20161207T102314_ch1-predicted-1.0pr.csv
       groundtruth-data/round2/20161207T102314_ch1-detected.csv
       groundtruth-data/round2/20161207T102314_ch1-missed.csv
       trained-classifier2/train_1r/thresholds.ckpt-10.csv
       trained-classifier2/train_1r/thresholds.ckpt-100.csv
       groundtruth-data/round1/PS_20130625111709_ch3-mistakes.csv
       omit-one/generalize_1w/thresholds.ckpt-10.csv
       omit-one/generalize_1w/thresholds.ckpt-100.csv
       nfeatures-32/xvalidate_1k/thresholds.ckpt-10.csv
       nfeatures-32/xvalidate_1k/thresholds.ckpt-100.csv
       nfeatures-64/xvalidate_1k/thresholds.ckpt-10.csv
       nfeatures-64/xvalidate_1k/thresholds.ckpt-100.csv
       groundtruth-data/congruence-tic.1.0pr.csv
       groundtruth-data/congruence-word.1.0pr.csv
      )
for file in ${files[*]} ; do
  if [[ $(diff $repo_path/test/scratch/sh/$file \
               $repo_path/test/scratch/py/$file) ]] ; then
      echo ERROR in $file
  fi
done

$repo_path/test/seeds.py