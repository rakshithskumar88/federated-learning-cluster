import torch

try:
    print("Is ROCm (hip) available?", torch.cuda.is_available())
    if torch.cuda.is_available():
        print(f"Device Name: {torch.cuda.get_device_name(0)}")
        # Try to allocate a small tensor
        x = torch.tensor([1.0, 2.0]).cuda()
        print("Tensor successfully moved to GPU:", x)
    else:
        print("ROCm is not detected by PyTorch.")
except Exception as e:
    print(f"Error: {e}")
