#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void multiply_low_rank_matrix_cuda_kernel(
    const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits> A,
    const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits> B,
    const torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits> vector,
    torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits> result) {

  const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ scalar_t intermediate_vector[1024]; // Shared memory for intermediate vector

  // Phase 1: Each thread computes one element of the intermediate vector (Bv)
  if(thread_id < B.size(0)) {
    intermediate_vector[thread_id] = 0;
    for (int j = 0; j < B.size(1); ++j) {
      intermediate_vector[thread_id] += B[thread_id][j] * vector[j];
    }
  }
  __syncthreads(); // Synchronize threads to ensure the intermediate vector is fully computed

  // Phase 2: Each thread computes one element of the final output vector (Av)
  if(thread_id < A.size(0)) {
    result[thread_id] = 0;
    for (int j = 0; j < A.size(1); ++j) {
      result[thread_id] += A[thread_id][j] * intermediate_vector[j];
    }
  }
}

void multiply_low_rank_matrix_cuda(torch::Tensor A, torch::Tensor B, torch::Tensor vector, torch::Tensor result) {
  const dim3 threads(1024);
  const dim3 blocks((A.size(0) + threads.x - 1) / threads.x);

  AT_DISPATCH_FLOATING_TYPES(A.type(), "multiply_low_rank_matrix_cuda_kernel", ([&] {
    multiply_low_rank_matrix_cuda_kernel<scalar_t><<<blocks, threads>>>(
        A.packed_accessor<scalar_t,2,torch::RestrictPtrTraits>(),
        B.packed_accessor<scalar_t,2,torch::RestrictPtrTraits>(),
        vector.packed_accessor<scalar_t,1,torch::RestrictPtrTraits>(),
        result.packed_accessor<scalar_t,1,torch::RestrictPtrTraits>());
  }));

  // Template free version
  //multiply_low_rank_matrix_cuda_kernel<<<blocks, threads>>>(
  //    A.packed_accessor<float,2,torch::RestrictPtrTraits>(),
  //    B.packed_accessor<float,2,torch::RestrictPtrTraits>(),
  //    vector.packed_accessor<float,1,torch::RestrictPtrTraits>(),
  //    result.packed_accessor<float,1,torch::RestrictPtrTraits>());
}
