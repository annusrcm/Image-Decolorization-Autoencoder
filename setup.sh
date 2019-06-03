#!/usr/bin/env bash

sudo apt-get install python-dev python3-dev
sudo apt-get install libmysqlclient-dev
sudo apt-get install python3.5-tk
requirements=$1
python3.5 -m pip install -r $requirements
sudo apt-get update