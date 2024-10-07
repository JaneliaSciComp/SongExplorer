#!/usr/bin/env zsh

SCRIPT_DIR=$( cd -- "$( dirname -- "${(%):-%N}" )" &> /dev/null && pwd )

export PATH=$SCRIPT_DIR/songexplorer/bin:$PATH

$SCRIPT_DIR/make-predictions.py $argv
