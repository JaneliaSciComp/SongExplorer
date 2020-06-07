#!/bin/bash

# generate Venn diagrams of false positives and false negatives
 
# congruence.sh <path-to-groundtruth> <wavfiles-with-dense-annotations-and-predictions> <parallelize>

# e.g.
# $DEEPSONG_BIN congruence.sh `pwd`/groundtruth-data PS_20130625111709_ch3_p2.wav,PS_20130625111709_ch3_p3.wav 1

data_dir=$1
wav_files=$2
parallelize=$3

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

expr="$DIR/congruence.py $data_dir $wav_files $parallelize"

cmd="date; hostname; $expr; sync; date"
echo $cmd

eval "$cmd"
