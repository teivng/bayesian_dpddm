#!/bin/bash

for i in $(seq 0 ${1}); do
    sbatch "experiments/s_scavenger.slrm"
done