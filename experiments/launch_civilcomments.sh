#!/bin/bash

for i in {0..49}; do
    sbatch experiments/sbatch_civilcomments.slrm
done