#!/bin/bash

for i in {0..500}; do
    sbatch "experiments/s_scavenger.slrm"
done
