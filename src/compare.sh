#!/bin/bash

# plot accuracy across different hyperparameter values
 
# compare.sh <config-file> <logdirs-prefix>

# e.g.
# deepsong compare.sh `pwd`/configuration.sh `pwd`/withheld

config_file=$1
logdirs_prefix=$2

source $config_file

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

expr="$DIR/compare.py $logdirs_prefix"

cmd="date; \
     hostname; \
     $expr; \
     date"
echo $cmd

logfile=$logdirs_prefix-compare.log
jobname=compare-$logdirs_prefix

compare_it "$cmd" "$logfile" "$jobname"
