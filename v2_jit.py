import torch
from torch.utils.cpp_extension import load

lowrank = load('lowrank', ['cublas3.cpp'], verbose=True)
print(dir(lowrank))

# Create some data
n, m, k, l = 5, 3, 3, 2
A1 = torch.randn(k, n-k, device='cuda')
LU = torch.randn(k, k, device='cuda')
A2 = torch.randn(m-k, k, device='cuda')

vector = torch.randn(l, m, device='cuda')

LU[:] = 1
vector[:] = 1

# Run the CUDA kernel
result = lowrank.multiply_low_rank_matrix(A1, LU, A2, vector)

print(result)
print()

# Verify the result
L = LU.tril(-1) + torch.eye(k, device='cuda')
U = LU.triu()
U2 = LU.triu(1) + torch.eye(k, device='cuda')
A = torch.cat([U2.T, A1], dim=1)
#B = torch.cat([U, A2], dim=0)
B = torch.cat([LU.tril().T, A2], dim=0)
expected = vector @ B @ A

print(f'{A=}')
print(f'{B=}')
print()
print(f'{U=}')
print(f'{vector[:, :k] @ U=}')
print(f'{vector[:, :k] @ U.T=}')
print(f'{vector[:, :k] @ LU.tril()=}')
print(f'{vector[:, :k] @ LU.tril().T=}')
print()
print(f'{vector @ B=}')
print(f'{vector @ B @ A}')

assert torch.allclose(result, expected)

print("Success!")
