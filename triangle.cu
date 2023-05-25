#include <torch/extension.h>

template <typename scalar_t>
__global__ void triangle_multiply_cuda_kernel(
    const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits> matrix,
    const torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits> vector,
    torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits> result) {

  // Get the index of the current thread
  const int64_t row = blockIdx.x * blockDim.x + threadIdx.x;

  // Ensure we are within bounds
  if (row < matrix.size(1)) {
    result[row] = 0;
    for (int64_t col = 0; col <= row; col++) {
      result[row] += matrix[row][col] * vector[col];
    }
  }
}

void triangle_multiply_cuda(torch::Tensor matrix, torch::Tensor vector, torch::Tensor result) {
  const int64_t threads = 256;
  const dim3 blocks((matrix.size(1) + threads - 1) / threads);

  AT_DISPATCH_FLOATING_TYPES(matrix.type(), "triangle_multiply_cuda", ([&] {
    triangle_multiply_cuda_kernel<scalar_t><<<blocks, threads>>>(
        matrix.packed_accessor<scalar_t,2,torch::RestrictPtrTraits>(),
        vector.packed_accessor<scalar_t,1,torch::RestrictPtrTraits>(),
        result.packed_accessor<scalar_t,1,torch::RestrictPtrTraits>());
  }));
}
