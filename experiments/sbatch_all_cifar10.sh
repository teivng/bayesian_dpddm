#!/bin/bash

# normal queue
for i in {0..3}; do
    sbatch "experiments/s_m0.slrm"
done
echo "Submitted s_m0.slrm"

# m queue
for i in {0..7}; do
    sbatch "experiments/s_m1.slrm"
done
echo "Submitted s_m1.slrm"

# m2 queue
for i in {0..11}; do
    sbatch "experiments/s_m2.slrm"
done
echo "Submitted s_m2.slrm"

# m3 queue
for i in {0..19}; do
    sbatch "experiments/s_m3.slrm"
done
echo "Submitted s_m3.slrm"

#m4 queue
for i in {0..31}; do
    sbatch "experiments/s_m4.slrm"
done
echo "Submitted s_m4.slrm"

#m5 queue
for i in {0..63}; do
    sbatch "experiments/s_m5.slrm"
done
echo "Submitted s_m5.slrm"

echo "All sbatched!" 