#!/bin/bash

CONFIG_NAME="camelyon17_best_50"
SAMPLE_SIZE=50

for SEED in $(seq 57 76); do
    echo "Running $CONFIG_NAME with seed=$SEED and sample_size=$SAMPLE_SIZE"
    python experiments/run.py --config-name=$CONFIG_NAME seed=$SEED dpddm.data_sample_size=$SAMPLE_SIZE
done
