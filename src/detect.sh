#!/bin/bash

# threshold an audio recording in both the time and frequency spaces
 
# detect.sh <config-file> <full-path-to-wavfile> <time-sigma> <time-smooth-ms> <frequency-n-ms> <frequency-nw> <frequency-p> <frequency-smooth-ms>

# e.g.
# deepsong detect.sh `pwd`/configuration.sh `pwd`/groundtruth-data/round1/20161207T102314_ch1_p1.wav 4 0.1

config_file=$1
filename=$2
time_sigma=$3
time_smooth_ms=$4
frequency_n_ms=$5
frequency_nw=$6
frequency_p=$7
frequency_smooth_ms=$8

source $config_file

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

expr="$DIR/detect.py $filename $tic_rate $time_sigma $time_smooth_ms $frequency_n_ms $frequency_nw $frequency_p $frequency_smooth_ms"

cmd="date; \
     hostname; \
     $expr; \
     date"
echo $cmd

jobname=detect-$filename
logfile=${filename%.wav}-detect.log

detect_it "$cmd" "$logfile" "$jobname"
