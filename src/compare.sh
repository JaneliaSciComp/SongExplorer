#!/bin/bash

# plot accuracy across different hyperparameter values
 
# compare.sh <logdirs-prefix>

# e.g.
# $SONGEXPLORER_BIN compare.sh `pwd`/withheld

logdirs_prefix=$1

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

echo SongExplorer version: $(cat $DIR/../VERSION.txt)

expr="$DIR/compare.py $logdirs_prefix"

cmd="date; hostname; $expr; sync; date"
echo $cmd

eval "$cmd"
