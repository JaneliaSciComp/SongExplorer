#!/bin/bash

# record whether annotations where correctly or mistakenly classified
 
# mistakes.sh <config-file> <path-to-annotations-npz-file>

# e.g.
# $DEEPSONG_BIN mistakes.sh `pwd`/configuration.sh `pwd`/groundtruth-data

config_file=$1
groundtruth_directory=$2

source $config_file

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

expr="$DIR/mistakes.py \"$groundtruth_directory\""

cmd="date; hostname; $expr; sync; date"
echo $cmd

logfile=${groundtruth_directory}/mistakes.log

mistakes_it "$cmd" "$logfile"
