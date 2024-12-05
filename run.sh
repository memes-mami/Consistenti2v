#!/bin/bash

# Ensure that the user has provided a prompt
if [ $# -eq 0 ]; then
    echo "Usage: $0 <prompt>"
    exit 1
fi


# Activate the conda environment
source ~/miniconda3/bin/activate consisti2v

# Run the Python script with the provided prompt
python test.py "$1"

# Deactivate the conda environment (optional)

source ~/miniconda3/bin/activate consisti2v2

python app2.py --prompt "$PROMPT" --input_image_path "generated_image.png"
