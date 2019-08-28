#!/bin/bash

# apply per-class thresholds to discretize probabilities

# ethogram.sh <config-file> <logdir> <model> <check-point> <tf-file>

# e.g.
# $DEEPSONG_BIN ethogram.sh `pwd`/configuration.sh `pwd`/trained-classifier 1k 50 `pwd`/groundtruth-data/round1/20161207T102314_ch1_p1.tf

config_file=$1
logdir=$2
model=$3
check_point=$4
tffile=$5

source $config_file

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

expr="$DIR/ethogram.py $logdir $model $check_point $tffile $audio_tic_rate"

cmd="date; hostname; $expr; sync; date"
echo $cmd

tfdir=`dirname $tffile`
tfbase=`basename $tffile`
logfile=$tfdir/${tfbase%.tf}-ethogram.log

ethogram_it "$cmd" "$logfile"
