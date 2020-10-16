#!/bin/bash

# threshold an audio recording in both the time and frequency spaces
 
# detect.sh <full-path-to-wavfile> <time-sigma-signal> <time-sigma-noise> <time-smooth-ms> <frequency-n-ms> <frequency-nw> <frequency-p-signal> <frequency-p-noise> <frequency-smooth-ms> <audio-tic-rate> <audio-nchannels>

# e.g.
# $SONGEXPLORER_BIN detect.sh `pwd`/groundtruth-data/round2/20161207T102314_ch1_p1.wav 4 2 6.4 25.6 4 0.1 1.0 25.6 2500 1

wavfile=$1
time_sigma_signal=$2
time_sigma_noise=$3
time_smooth_ms=$4
frequency_n_ms=$5
frequency_nw=$6
frequency_p_signal=$7
frequency_p_noise=$8
frequency_smooth_ms=$9
audio_tic_rate=${10}
audio_nchannels=${11}

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

expr="$DIR/detect.py $wavfile $audio_tic_rate $audio_nchannels $time_sigma_signal $time_sigma_noise $time_smooth_ms $frequency_n_ms $frequency_nw $frequency_p_signal $frequency_p_noise $frequency_smooth_ms"

cmd="date; hostname; $expr; sync; date"
echo $cmd

eval "$cmd"
