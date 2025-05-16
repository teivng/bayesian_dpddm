#!/bin/bash

CONFIG_NAME="camelyon17_best_10"
SAMPLE_SIZE=10

for SEED in $(seq 86 95); do
    echo "Running $CONFIG_NAME with seed=$SEED and sample_size=$SAMPLE_SIZE"
    python experiments/run.py --config-name=$CONFIG_NAME seed=$SEED dpddm.data_sample_size=$SAMPLE_SIZE
done
