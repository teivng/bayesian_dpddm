#!/bin/bash

find experiments/ -type f -name "*.slrm" -exec sed -i 's/^wandb agent .*/$1' {} +