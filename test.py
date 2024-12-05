import argparse
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
import torch
from PIL import Image
import matplotlib.pyplot as plt

# Parse the command line argument for the prompt
parser = argparse.ArgumentParser()
parser.add_argument("prompt", type=str, help="The prompt for the image generation")
args = parser.parse_args()

# Load the pipeline with float32 precision
repo_id = "stabilityai/stable-diffusion-2-base"
pipe = DiffusionPipeline.from_pretrained(repo_id, torch_dtype=torch.float32)

# Switch to a GPU-based scheduler
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")

# Increase the resolution by setting height and width
high_res_image = pipe(args.prompt, num_inference_steps=50, height=512, width=512).images[0]

# Display the generated high-resolution image
plt.imshow(high_res_image)
plt.axis('off')  # Hide axes
plt.show()

# Save the image as a .png file
high_res_image.save("generated_image.png", format="PNG")
