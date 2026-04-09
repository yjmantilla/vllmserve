import torch
import gc

# Delete any model/tensor references
# del model  # or whatever variable holds the model
# del tensor

# Force garbage collection
gc.collect()

# Free the cached memory PyTorch is holding
torch.cuda.empty_cache()
