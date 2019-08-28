#!/bin/bash

# generate Venn diagrams of false positives and false negatives
 
# congruence.sh <config-file> <path-to-groundtruth> <wavfiles-with-dense-annotations-and-predictions>

# e.g.
# $DEEPSONG_BIN congruence.sh `pwd`/configuration.sh `pwd`/groundtruth-data PS_20130625111709_ch3_p2.wav,PS_20130625111709_ch3_p3.wav

config_file=$1
data_dir=$2
wav_files=$3

source $config_file

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

expr="$DIR/congruence.py $data_dir $wav_files $congruence_parallelize"

cmd="date; hostname; $expr; sync; date"
echo $cmd

logfile=$data_dir/congruence.log

congruence_it "$cmd" "$logfile"
