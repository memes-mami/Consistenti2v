import argparse
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
import torch
from PIL import Image
import matplotlib.pyplot as plt

# Parse the command line argument for the prompt
parser = argparse.ArgumentParser(description="Generate an image based on a text prompt.")
parser.add_argument("prompt", type=str, help="The prompt for the image generation")
args = parser.parse_args()

# Load the DiffusionPipeline with DPMSolverMultistepScheduler for faster inference
pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0")
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.to("cuda" if torch.cuda.is_available() else "cpu")

# Generate the image using the provided prompt
image = pipe(args.prompt).images[0]

# Save and display the generated image
image.save("generated_image.png")
plt.imshow(image)
plt.axis("off")
plt.show()  # Display the image in a pop-up window
