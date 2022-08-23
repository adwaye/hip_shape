#!/bin/bash
rm -r ./viewboth
python3 initial_data_analysis.py
tensorboard --logdir=./view

