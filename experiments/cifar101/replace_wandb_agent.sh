#!/bin/bash

if [ -z "$1" ]; then
  echo "Usage: $0 <replacement_string>"
  exit 1
fi

find experiments/cifar101/ -type f -name "*.slrm" -exec sed -i "s/^wandb agent .*/$(echo "$1" | sed 's/[\/&]/\\&/g')/" {} +