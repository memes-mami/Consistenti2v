import torch

# Clears the cache
torch.cuda.empty_cache()

# Frees any cached memory not being used
torch.cuda.memory_reserved(0)
torch.cuda.memory_allocated(0)
