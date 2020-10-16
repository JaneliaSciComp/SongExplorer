#!/bin/bash

# generate Venn diagrams of false positives and false negatives
 
# congruence.sh <path-to-groundtruth> <wavfiles-with-dense-annotations-and-predictions> <nprobabilities> <audio-tic-rate> <parallelize>

# e.g.
# $SONGEXPLORER_BIN congruence.sh `pwd`/groundtruth-data PS_20130625111709_ch3_p2.wav,PS_20130625111709_ch3_p3.wav 20 2500 1

data_dir=$1
wav_files=$2
nprobabilities=$3
audio_tic_rate=$4
parallelize=$5

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

expr="$DIR/congruence.py $data_dir $wav_files $nprobabilities $audio_tic_rate $parallelize"

cmd="date; hostname; $expr; sync; date"
echo $cmd

eval "$cmd"
