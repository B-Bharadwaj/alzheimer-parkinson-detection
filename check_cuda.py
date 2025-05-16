import torch

print("🧠 PyTorch CUDA Check")
print("---------------------")
print(f"Is CUDA available?        {torch.cuda.is_available()}")
print(f"Number of GPUs available: {torch.cuda.device_count()}")

if torch.cuda.is_available():
    print(f"GPU in use:               {torch.cuda.get_device_name(0)}")
    x = torch.rand(1000, 1000).to('cuda')
    y = torch.mm(x, x)
    print(f"Test matrix multiplied on: {x.device}")
else:
    print("⚠️ CUDA not available. Running on CPU.")
