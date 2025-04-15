#!/bin/bash

for i in {0..20}; do
    sbatch experiments/sbatch_camelyon17.slrm
done