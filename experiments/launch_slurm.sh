#!/bin/bash

if [ -z "$1" ] || [ -z "$2" ] || [ -z "$3" ]; then
  echo "Usage: $0 <replacement_string> <run_count_per_job> <num_of_scavengers>"
  exit 1
fi

# Extract the part after "wandb agent" from the replacement string
wandb_agent_part=$(echo "$1" | sed 's/^wandb agent //')

# Escape the extracted part for use in sed
escaped_wandb_agent_part=$(echo "$wandb_agent_part" | sed 's/[\/&]/\\&/g')

# Construct the new replacement string with the --count flag
replacement_string_with_count="wandb agent --count $2 $escaped_wandb_agent_part"

# Find and replace the line starting with "wandb agent" in all .slrm files
find experiments/slrm -type f -name "*.slrm" -exec sed -i "s/^wandb agent .*/$replacement_string_with_count/" {} +

# normal queue
for i in {0..3}; do
    sbatch "experiments/slrm/s_m0.slrm"
done
echo "Submitted s_m0.slrm"

# m queue
for i in {0..7}; do
    sbatch "experiments/slrm/s_m1.slrm"
done
echo "Submitted s_m1.slrm"

# m2 queue
for i in {0..11}; do
    sbatch "experiments/slrm/s_m2.slrm"
done
echo "Submitted s_m2.slrm"

# m3 queue
for i in {0..19}; do
    sbatch "experiments/slrm/s_m3.slrm"
done
echo "Submitted s_m3.slrm"

#m4 queue
for i in {0..31}; do
    sbatch "experiments/slrm/s_m4.slrm"
done
echo "Submitted s_m4.slrm"

#m5 queue
for i in {0..63}; do
    sbatch "experiments/slrm/s_m5.slrm"
done
echo "Submitted s_m5.slrm"

for i in $(seq 0 ${3}); do
    sbatch "experiments/s_scavenger.slrm"
done

