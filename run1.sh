#!/bin/bash

# Ensure that the user has provided a prompt
if [ $# -eq 0 ]; then
    echo "Usage: $0 <prompt>"
    exit 1
fi


# Activate the conda environment
source ~/miniconda3/bin/activate consisti2v

# Run test3.py and capture the output filename
output_file=$(python test3.py "$1" | tail -n 1)

# Activate the conda environment for app2.py
source ~/miniconda3/bin/activate consisti2v2

# Run app2.py using the generated filename as input
python app2.py --prompt "$1" --input_image_path "$output_file"
