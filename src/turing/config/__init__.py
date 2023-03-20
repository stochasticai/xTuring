import torch

# check if cuda is available, if not use cpu and throw warning
DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_DTYPE = torch.float16 if DEFAULT_DEVICE.type == "cuda" else torch.float32


if DEFAULT_DEVICE.type == "cpu":
    print("WARNING: CUDA is not available, using CPU instead, can be very slow")
