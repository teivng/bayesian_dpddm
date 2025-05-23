#!/bin/bash

CONFIG_NAME="camelyon17_best"
SAMPLE_SIZE=100

for SEED in $(seq 67 76); do
    echo "Running $CONFIG_NAME with seed=$SEED and sample_size=$SAMPLE_SIZE"
    python experiments/run.py --config-name=$CONFIG_NAME seed=$SEED dpddm.data_sample_size=$SAMPLE_SIZE
done
