#!/bin/bash

# generate Venn diagrams of false positives and false negatives
 
# congruence.sh <path-to-groundtruth> <wavfiles-with-dense-annotations-and-predictions> <portion> <convolve-ms> <nprobabilities> <audio-tic-rate> <parallelize>

# e.g.
# $SONGEXPLORER_BIN congruence.sh `pwd`/groundtruth-data PS_20130625111709_ch3_p2.wav,PS_20130625111709_ch3_p3.wav 1 0 20 2500 1

data_dir=$1
wav_files=$2
portion=$3
convolve_ms=$4
nprobabilities=$5
audio_tic_rate=$6
parallelize=$7

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

expr="$DIR/congruence.py $data_dir $wav_files $portion $convolve_ms $nprobabilities $audio_tic_rate $parallelize"

cmd="date; hostname; $expr; sync; date"
echo $cmd

eval "$cmd"
