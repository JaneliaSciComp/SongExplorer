#!/bin/bash

# generate .wav files of per-class probabilities from a .tf log file

# classify2.sh <context_ms> <shiftby_ms> <representation> <stride_ms> <logdir> <model> <check-point> <wavfile> <audio-tic-rate> [<labels> <prevalences>]

# e.g.
# $DEEPSONG_BIN classify2.sh 204.8 0.0 waveform 1.6 `pwd`/trained-classifier train_1 50 `pwd`/groundtruth-data/round1/20161207T102314_ch1_p1.wav 5000 mel-pulse,mel-sine,ambient 0.1,0.1,0.8

context_ms=$1
shiftby_ms=$2
representation=$3
stride_ms=$4
logdir=$5
model=$6
check_point=$7
wavfile=$8
audio_tic_rate=$9
dummy_placeholder=${10}
if [[ $# == 12 ]] ; then
  labels=${11}
  prevalences=${12}
else
  labels=
  prevalences=
fi

if [ "$representation" == "waveform" ] ; then
  stride_ms=`dc -e "16 k 1000 $audio_tic_rate / p"`
fi
frozenlog=$logdir/$model/frozen-graph.ckpt-${check_point}.log
ndownsample2=`grep -e 'strides = \[1, 2' -e 'strides = 2' $frozenlog | wc -l`
if (( "$ndownsample2" > 0 )) ; then
  stride_ms=`dc -e "$stride_ms 2 $ndownsample2 ^ * p"`
fi

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

wavdir=`dirname $wavfile`
wavbase=`basename $wavfile`
tffile=$wavdir/${wavbase%.*}.tf

expr="$DIR/classify2.py $logdir $model $tffile $context_ms $shiftby_ms $stride_ms $labels $prevalences"

cmd="date; hostname; $expr; sync; date"
echo $cmd

eval "$cmd"
