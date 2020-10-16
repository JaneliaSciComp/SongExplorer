#!/bin/bash

# apply per-class thresholds to discretize probabilities

# ethogram.sh <logdir> <model> <thresholds_file> <tf-file> <audio-tic_rate>

# e.g.
# $SONGEXPLORER_BIN ethogram.sh `pwd`/trained-classifier 1k 50 `pwd`/groundtruth-data/round1/20161207T102314_ch1_p1.tf 5000

logdir=$1
model=$2
thresholds_file=$3
tf_file=$4
audio_tic_rate=$5

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

expr="$DIR/ethogram.py $logdir $model $thresholds_file $tf_file $audio_tic_rate"

cmd="date; hostname; $expr; sync; date"
echo $cmd

eval "$cmd"
