import torch
from torch.utils.cpp_extension import load

lowrank = load('lowrank', ['lowrank.cpp', 'lowrank.cu'], verbose=True)
print(dir(lowrank))

# Create some data
n, m, k = 5, 4, 3
A = torch.randn(n, k, device='cuda')
B = torch.randn(k, m, device='cuda')
vector = torch.randn(m, device='cuda')

# Run the CUDA kernel
result = lowrank.multiply_low_rank_matrix(A, B, vector)

# Verify the result
expected = A @ B @ vector

assert torch.allclose(result, expected)

print("Success!")
