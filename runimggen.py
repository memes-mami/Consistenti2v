import argparse
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from datetime import datetime
import os

# Set up argument parsing
parser = argparse.ArgumentParser(description="Generate an image based on a text prompt.")
parser.add_argument("prompt", type=str, help="The prompt for the image generation")
parser.add_argument("--output", type=str, default="output_image.png", help="Filename or path to save the generated image")
args = parser.parse_args()

# Load the DiffusionPipeline and configure the scheduler for faster inference
pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0")
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

# Generate image based on the prompt
image = pipe(args.prompt).images[0]

# Prepare the output path and filename
output_dir = os.path.dirname(args.output)
if output_dir and not os.path.exists(output_dir):
    os.makedirs(output_dir)  # Create directory if it doesn't exist

# Use timestamp to create a unique filename if the file already exists
if os.path.exists(args.output):
    base, ext = os.path.splitext(args.output)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.output = f"{base}_{timestamp}{ext}"

# Save the image
image.save(args.output)
print(f"Image generated and saved as {args.output}")
print(f"{args.output}")
