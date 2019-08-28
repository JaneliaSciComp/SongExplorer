#!/bin/bash

# generate Venn diagrams of false positives and negatives
 
# dense.py <config-file> <folder-with-dense-annotations-and-predictions>

# e.g.
# deepsong dense `pwd`/configuration.sh `pwd`/groundtruth-folder/test-folder

config_file=$1
test_folder=$2

source $config_file

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

expr="$DIR/dense.py $test_folder"

cmd="date; \
     hostname; \
     $expr; \
     date"
echo $cmd

logfile=${test_folder}/dense.log
jobname=dense-$test_folder

dense_it "$cmd" "$logfile" "$jobname"
