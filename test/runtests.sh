#!/bin/bash

# run all of the tests

# ${SONGEXPLORER_BIN/-B/-B /tmp:/opt/songexplorer/test/scratch -B} bash -c "test/runtests.sh"

repo_path=$(dirname $(dirname $(which detect.sh)))

$repo_path/test/tutorial.sh
$repo_path/test/tutorial.py

if (( $(diff <(tree $repo_path/test/scratch/tutorial-sh | grep -v tfevents) \
             <(tree $repo_path/test/scratch/tutorial-py | grep -v tfevents) | wc -l) > 4 )) ; then
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
       groundtruth-data/congruence.tic.ambient.csv
       groundtruth-data/congruence.tic.mel-pulse.csv
       groundtruth-data/congruence.tic.mel-sine.csv
       groundtruth-data/congruence.word.ambient.csv
       groundtruth-data/congruence.word.mel-pulse.csv
       groundtruth-data/congruence.word.mel-sine.csv
      )
for file in ${files[*]} ; do
  if [[ $(diff $repo_path/test/scratch/tutorial-sh/$file \
               $repo_path/test/scratch/tutorial-py/$file) ]] ; then
      echo WARNING $file in tutorial-sh/ and tutorial-py/ differ
  fi
done

$repo_path/test/seeds.py
$repo_path/test/freeze-classify.py
$repo_path/test/annotating.py
$repo_path/test/congruence.py
$repo_path/test/shiftby.sh ; $repo_path/test/shiftby.py
