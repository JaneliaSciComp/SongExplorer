#!/usr/bin/env zsh

SCRIPT_DIR=$( cd -- "$( dirname -- "${(%):-%N}" )" &> /dev/null && pwd )

source $SCRIPT_DIR/songexplorer/bin/activate

$SCRIPT_DIR/songexplorer/bin/songexplorer/src/songexplorer $SCRIPT_DIR/configuration.py 8080

sleep 10
