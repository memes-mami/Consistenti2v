from diffusers import DiffusionPipeline
import torch
from PIL import Image

# Load the pipeline and move it to CUDA if available
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0").to(device)

prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
# Generate the image (making sure to specify the device for computation)
image = pipe(prompt).images[0]

# Save and view the image
image.save("astronaut_jungle.png")
image.show()  # Opens the image in the default image viewer on your system
