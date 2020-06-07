#!/bin/bash

# find events that are detected but not predicted, a.k.a. false negatives
 
# misses.sh <detected-and-predicted-csv-files...>

# e.g.
# $DEEPSONG_BIN misses.sh `pwd`/groundtruth-data/round2/20161207T102314_ch1_p2-detected.csv,`pwd`/groundtruth-data/round2/20161207T102314_ch1_p2-predicted-1.0pr.csv

IFS=',' read -ra csv_files <<< "$1"

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

expr="$DIR/misses.py $directory ${csv_files[@]}"

cmd="date; hostname; $expr; sync; date"
echo $cmd

eval "$cmd"
