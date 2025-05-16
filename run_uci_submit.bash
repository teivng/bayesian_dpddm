#!/bin/bash

# run_cifar_submit.bash
# Runs the experiment 30 times with seeds from 57 to 86

for ((i=0; i<30; i++)); do
    seed=$((57 + i))
    echo "Running with seed=$seed"
    python experiments/run.py --config-name=uci_best seed=$seed
done
