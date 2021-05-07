#!/bin/bash

# generate .wav files of per-class probabilities from a .tf log file

# classify2.sh <context_ms> <shiftby_ms> <logdir> <model> <check-point> <wavfile> <audio-tic-rate> [<labels> <prevalences>]

# e.g.
# $SONGEXPLORER_BIN classify2.sh 204.8 0.0 `pwd`/trained-classifier train_1 50 `pwd`/groundtruth-data/round1/20161207T102314_ch1_p1.wav 5000 mel-pulse,mel-sine,ambient 0.1,0.1,0.8

context_ms=$1
shiftby_ms=$2
logdir=$3
model=$4
check_point=$5
wavfile=$6
audio_tic_rate=$7
dummy_placeholder=$8
if [[ $# == 10 ]] ; then
  labels=$9
  prevalences=${10}
else
  labels=
  prevalences=
fi

frozenlog=$logdir/$model/frozen-graph.ckpt-${check_point}.log

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

echo SongExplorer version: $(cat $DIR/../VERSION.txt)

wavdir=`dirname $wavfile`
wavbase=`basename $wavfile`
tffile=$wavdir/${wavbase%.*}.tf

expr="$DIR/classify2.py $logdir $model $tffile $context_ms $shiftby_ms $labels $prevalences"

cmd="date; hostname; $expr; sync; date"
echo $cmd

eval "$cmd"
