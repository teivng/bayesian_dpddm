#!/bin/bash

# Define the expected directory name
EXPECTED_DIR="bayesian_dpddm"

# Get the current working directory's basename
CURRENT_DIR=$(basename "$PWD")

# Check if the current directory matches the expected directory
if [ "$CURRENT_DIR" != "$EXPECTED_DIR" ]; then
  echo "Error: This script must be run from the '$EXPECTED_DIR' directory."
  exit 1
fi

# Delete the *.egg-info/ directories
rm -rf ./*.egg-info/

# Delete everything in dist/* and wandb/*
rm -rf dist/* wandb/*

echo "Cleanup completed successfully."
