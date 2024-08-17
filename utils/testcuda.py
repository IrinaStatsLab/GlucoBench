import torch

if torch.cuda.is_available():
    print("CUDA is available! You can use GPU for deep learning.")
else:
    print("CUDA is not available. You will be using CPU for deep learning.")