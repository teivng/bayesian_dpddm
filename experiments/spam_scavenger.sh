#!/bin/bash

for i in {0..199}; do
    sbatch "experiments/s_scavenger.slrm"
done