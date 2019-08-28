#!/bin/bash

# find events that are detected but not predicted, a.k.a. false negatives
 
# misses.sh <config-file> <detected-and-predicted-csv-files...>

# e.g.
# deepsong misses.sh `pwd`/configuration.sh `pwd`/groundtruth-data/round2/20161207T102314_ch1_p2-detected.csv,`pwd`/groundtruth-data/round2/20161207T102314_ch1_p2-predicted-1.0pr.csv

config_file=$1
IFS=',' read -ra csv_files <<< "$2"

source $config_file

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

expr="$DIR/misses.py $directory ${csv_files[@]}"

cmd="date; \
     hostname; \
     $expr; \
     date"
echo $cmd

jobname=misses-$filename

csvfile0=${csv_files[0]}
firstline=$(head -n 1 $csvfile0)
wavfile=${firstline%%,*}
logfile=$(dirname $csvfile0)/${wavfile%.wav}-missed.log

misses_it "$cmd" "$logfile" "$jobname"
