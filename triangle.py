import torch
from torch.utils.cpp_extension import load_inline

# Define CUDA kernel
cuda_source = """
#include <torch/extension.h>

extern "C" {
__global__ void triangle_multiply_cuda(
    const float* __restrict__ matrix,
    const float* __restrict__ vector,
    float* __restrict__ result,
    size_t n) {

  const int64_t column = blockIdx.x * blockDim.x + threadIdx.x;

  if (column < n) {
    result[column] = 0;
    for (int64_t row = 0; row <= column; ++row) {
      result[column] += matrix[row * n + column] * vector[row];
    }
  }
}
}
"""

# Load CUDA kernel
triangle_multiply_cuda = load_inline(
        name='triangle_multiply_cuda',
        cuda_sources=[cuda_source],
        cpp_sources=[],
        verbose=True,
        extra_cuda_cflags=["-gencode", "arch=compute_75,code=sm_75"],  # Replace with your GPU architecture
        functions='triangle_multiply_cuda'
        )

#>>> from torch.utils.cpp_extension import load_inline
#>>> source = """
#at::Tensor sin_add(at::Tensor x, at::Tensor y) {
#  return x.sin() + y.sin();
#}
#"""
#>>> module = load_inline(name='inline_extension',
#...                      cpp_sources=[source],
#...                      functions=['sin_add'])


# Create some data
n = 1000
matrix = torch.randn(n, n, device='cuda').tril()
vector = torch.randn(n, device='cuda')
result = torch.empty(n, device='cuda')

# Calculate grid and block dimensions
threads = 256
blocks = (n + threads - 1) // threads

# Run the CUDA kernel
triangle_multiply_cuda(grid=(blocks,), block=(threads,), args=[matrix.data_ptr(), vector.data_ptr(), result.data_ptr(), n])

# Verify the result
expected = torch.mv(matrix, vector)
assert torch.allclose(result, expected)
