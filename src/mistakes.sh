#!/bin/bash

# record whether annotations where correctly or mistakenly classified
 
# mistakes.sh <path-to-annotations-npz-file>

# e.g.
# $SONGEXPLORER_BIN mistakes.sh `pwd`/groundtruth-data

groundtruth_directory=$1

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

echo SongExplorer version: $(cat $DIR/../VERSION.txt)

expr="$DIR/mistakes.py \"$groundtruth_directory\""

cmd="date; hostname; $expr; sync; date"
echo $cmd

eval "$cmd"
