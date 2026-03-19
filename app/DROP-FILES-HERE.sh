#!/usr/bin/env zsh

SCRIPT_DIR=$( cd -- "$( dirname -- "${(%):-%N}" )" &> /dev/null && pwd )

source $SCRIPT_DIR/songexplorer/bin/activate

$SCRIPT_DIR/make-predictions.py $argv

sleep 10
