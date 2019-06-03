#!/usr/bin/env bash

BASEDIR=$(dirname "$0")

python3 $BASEDIR/preprocess_data.py
python3 $BASEDIR/train_and_test.py