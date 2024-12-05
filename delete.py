import os
import shutil

# Define the Hugging Face cache directory
hf_cache_dir = os.path.expanduser("~/.cache/huggingface/hub")

# Check if the directory exists
if os.path.exists(hf_cache_dir):
    print(f"Deleting Hugging Face models from: {hf_cache_dir}")
    
    # Use shutil.rmtree to delete the directory and its contents
    try:
        shutil.rmtree(hf_cache_dir)
        print("All Hugging Face models have been deleted.")
    except Exception as e:
        print(f"An error occurred while deleting the Hugging Face cache: {e}")
else:
    print("Hugging Face cache directory does not exist. Nothing to delete.")
