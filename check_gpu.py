"""Quick check that PyTorch can use the GPU. Run: .venv\\Scripts\\python check_gpu.py"""
import sys

import torch

print("Python      :", sys.version.split()[0])
print("PyTorch     :", torch.__version__, "| CUDA build:", torch.version.cuda)
print("CUDA ready  :", torch.cuda.is_available())
if not torch.cuda.is_available():
    sys.exit("No GPU found. Install the cu128 build of torch (see README.md).")

print("GPU         :", torch.cuda.get_device_name(0))
print("Compute cap :", "sm_%d%d" % torch.cuda.get_device_capability(0))
print("GPU memory  : %.1f GB" % (torch.cuda.get_device_properties(0).total_memory / 1e9))

# A real kernel launch: fails with "no kernel image is available" on the wrong build.
a = torch.randn(1024, 1024, device="cuda")
b = a @ a.T
conv = torch.nn.Conv2d(3, 8, 3).cuda()
with torch.autocast("cuda", dtype=torch.float16):
    y = conv(torch.randn(2, 3, 64, 64, device="cuda"))
torch.cuda.synchronize()
print("Test ops    : OK (matmul sum = %.1f, conv out = %s)" % (b.sum().item(), tuple(y.shape)))
