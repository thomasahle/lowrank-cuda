from torch.utils.cpp_extension import load
triangle = load(
    'triangle', ['triangle.cpp', 'triangle.cu'], verbose=True)
print(dir(triangle))

# Create some data
import torch
n = 5
matrix = torch.randn(n, n, device='cuda').tril()
vector = torch.randn(n, device='cuda')
result = torch.empty(n, device='cuda')

# Calculate grid and block dimensions
threads = 256
blocks = (n + threads - 1) // threads

# Run the CUDA kernel
#triangle.triangle_multiply(grid=(blocks,), block=(threads,), args=[matrix.data_ptr(), vector.data_ptr(), result.data_ptr(), n])
triangle.triangle_multiply_cuda(matrix, vector, result)

print(matrix)
print(vector)
print(result)

# Verify the result
expected = matrix @ vector

print(expected)

assert torch.allclose(result, expected)
