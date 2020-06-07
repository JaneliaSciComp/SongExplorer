#!/bin/bash

# generate confusion matrices, precision-recall curves, thresholds, etc.
 
# accuracy.sh <logdir> <precision-recall-ratios> <n-probabilities> <parallelize>

# e.g.
# $DEEPSONG_BIN accuracy.sh `pwd`/trained-classifier 2,1,0.5 50 1

logdir=$1
precision_recall_ratios=$2
nprobabilities=$3
parallelize=$4

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

expr="$DIR/accuracy.py $logdir $precision_recall_ratios $nprobabilities $parallelize"

cmd="date; hostname; $expr; sync; date"
echo $cmd

eval "$cmd"
