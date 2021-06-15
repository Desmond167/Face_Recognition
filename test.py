import torch

flag_gpu_available = torch.cuda.is_available()
print(flag_gpu_available)