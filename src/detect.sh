#!/bin/bash

# threshold an audio recording in both the time and frequency spaces
 
# detect.sh <full-path-to-wavfile> <time-sigma> <time-smooth-ms> <frequency-n-ms> <frequency-nw> <frequency-p> <frequency-smooth-ms> <audio-tic-rate> <audio-nchannels>

# e.g.
# $DEEPSONG_BIN detect.sh `pwd`/groundtruth-data/round1/20161207T102314_ch1_p1.wav 6 6.4 25.6 4 0.1 25.6 5000 1

wavfile=$1
time_sigma=$2
time_smooth_ms=$3
frequency_n_ms=$4
frequency_nw=$5
frequency_p=$6
frequency_smooth_ms=$7
audio_tic_rate=$8
audio_nchannels=$9

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

expr="$DIR/detect.py $wavfile $audio_tic_rate $audio_nchannels $time_sigma $time_smooth_ms $frequency_n_ms $frequency_nw $frequency_p $frequency_smooth_ms"

cmd="date; hostname; $expr; sync; date"
echo $cmd

eval "$cmd"
