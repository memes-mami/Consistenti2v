import os
import json
import torch
import random
import requests
from PIL import Image
import numpy as np
from datetime import datetime
import argparse

import torchvision.transforms as T
from diffusers import DDIMScheduler
from diffusers.utils.import_utils import is_xformers_available
from consisti2v.pipelines.pipeline_conditional_animation import ConditionalAnimationPipeline
from consisti2v.utils.util import save_videos_grid
from omegaconf import OmegaConf

# Force device to CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
assert device.type == "cuda", "CUDA is not available. Please check your GPU setup."

class AnimateController:
    def __init__(self):
        # Configuration directories
        self.basedir = os.getcwd()
        self.savedir = os.path.join(self.basedir, "generated_videos", datetime.now().strftime("%Y-%m-%dT%H-%M-%S"))
        os.makedirs(self.savedir, exist_ok=True)
        
        # Default resolution
        self.image_resolution = (256, 256)
        
        # Load pipeline and set to CUDA
        self.pipeline = ConditionalAnimationPipeline.from_pretrained("TIGER-Lab/ConsistI2V", torch_dtype=torch.float16)
        self.pipeline.to(device)

    def load_and_resize_image(self, input_image_path, width, height, center_crop=False):  # Set center_crop default to False
        if input_image_path.startswith("http://") or input_image_path.startswith("https://"):
            image = Image.open(requests.get(input_image_path, stream=True).raw).convert("RGB")
        else:
            image = Image.open(input_image_path).convert("RGB")

        if center_crop:
            aspect_ratio = image.width / image.height
            crop_aspect_ratio = width / height
            if aspect_ratio > crop_aspect_ratio:
                new_width = int(crop_aspect_ratio * image.height)
                left = (image.width - new_width) // 2
                image = image.crop((left, 0, left + new_width, image.height))
            elif aspect_ratio < crop_aspect_ratio:
                new_height = int(image.width / crop_aspect_ratio)
                top = (image.height - new_height) // 2
                image = image.crop((0, top, image.width, top + new_height))

        return image.resize((width, height))

    def animate(self, prompt, negative_prompt, input_image_path, sample_steps=100, width=256, height=256,  # Updated default size to 256x256
                txt_cfg_scale=7.5, img_cfg_scale=1.0, center_crop=False, frame_stride=3, use_frameinit=True, 
                frameinit_noise_level=850, seed=None):
        
        # Set seed if provided, otherwise use random seed
        if seed is not None:
            torch.manual_seed(seed)
        else:
            seed = random.randint(0, 99999)
            torch.manual_seed(seed)
        
        # Load and preprocess the input image
        first_frame = self.load_and_resize_image(input_image_path, width, height, center_crop)
        img_transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        first_frame = img_transform(first_frame).unsqueeze(0).to(device)

        # Enable memory efficient attention if available
        if is_xformers_available() and int(torch.__version__.split(".")[0]) < 2:
            self.pipeline.unet.enable_xformers_memory_efficient_attention()

        # Optional: Initialize frame filtering
        if use_frameinit:
            self.pipeline.init_filter(
                width=width,
                height=height,
                video_length=16,
                filter_params=OmegaConf.create({'method': 'gaussian', 'd_s': 0.25, 'd_t': 0.25})
            )

        # Generate animation
        sample = self.pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            first_frames=first_frame,
            num_inference_steps=sample_steps,
            guidance_scale_txt=txt_cfg_scale,
            guidance_scale_img=img_cfg_scale,
            width=width,
            height=height,
            video_length=16,
            noise_sampling_method="pyoco_mixed",
            noise_alpha=1.0,
            frame_stride=frame_stride,
            use_frameinit=use_frameinit,
            frameinit_noise_level=frameinit_noise_level,
            camera_motion=None,
        ).videos

        # Save output video
        save_path = os.path.join(self.savedir, f"generated_video_{datetime.now().strftime('%H-%M-%S')}.mp4")
        save_videos_grid(sample, save_path, format="mp4")

        # Save config as JSON
        sample_config = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "first_frame_path": input_image_path,
            "num_inference_steps": sample_steps,
            "guidance_scale_text": txt_cfg_scale,
            "guidance_scale_image": img_cfg_scale,
            "width": width,
            "height": height,
            "seed": seed
        }
        config_path = os.path.join(self.savedir, "config.json")
        with open(config_path, "a") as config_file:
            json.dump(sample_config, config_file, indent=4)

        print(f"Video saved at: {save_path}")
        print(f"Configuration saved at: {config_path}")

# Main function to handle user inputs
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate video animation based on user prompt.")
    parser.add_argument("--prompt", type=str, required=True, help="Enter the text prompt for the animation.")
    parser.add_argument("--negative_prompt", type=str, default="", help="Enter the negative prompt to avoid in the animation.")
    parser.add_argument("--input_image_path", type=str, required=True, help="Path to the input image.")
    parser.add_argument("--sample_steps", type=int, default=100, help="Number of sampling steps.")  # Default to 100
    parser.add_argument("--width", type=int, default=256, help="Width of the generated video.")     # Default to 256
    parser.add_argument("--height", type=int, default=256, help="Height of the generated video.")   # Default to 256
    parser.add_argument("--txt_cfg_scale", type=float, default=7.5, help="Text guidance scale.")
    parser.add_argument("--img_cfg_scale", type=float, default=1.0, help="Image guidance scale.")
    parser.add_argument("--center_crop", action="store_true", help="Apply center cropping to the input image.")  # Removed default
    parser.add_argument("--frame_stride", type=int, default=3, help="Frame stride for the animation.")
    parser.add_argument("--use_frameinit", action="store_true", help="Use frame initialization.")
    parser.add_argument("--frameinit_noise_level", type=int, default=850, help="Noise level for frame initialization.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")

    args = parser.parse_args()

    controller = AnimateController()
    controller.animate(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        input_image_path=args.input_image_path,
        sample_steps=args.sample_steps,
        width=args.width,
        height=args.height,
        txt_cfg_scale=args.txt_cfg_scale,
        img_cfg_scale=args.img_cfg_scale,
        center_crop=args.center_crop,
        frame_stride=args.frame_stride,
        use_frameinit=args.use_frameinit,
        frameinit_noise_level=args.frameinit_noise_level,
        seed=args.seed
    )

