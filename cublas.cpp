#include <torch/extension.h>
#include <ATen/ATen.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

at::Tensor multiply_low_rank_matrix(at::Tensor A, at::Tensor B, at::Tensor v) {
  // Ensure the input tensors are on the same device and have the correct dimensionality
  TORCH_CHECK(A.device() == B.device() && B.device() == v.device(), "All tensors must be on the same device");
  TORCH_CHECK(A.dim() == 2 && B.dim() == 2 && v.dim() == 1, "Tensors must have correct dimensions: A(2D), B(2D), v(1D)");

  // Set up cuBLAS context
  cublasHandle_t handle;
  cublasCreate(&handle);

  // Ensure we're using the correct device
  cudaSetDevice(A.device().index());

  at::Tensor Bv = at::empty({B.size(0)}, B.options());
  float alpha = 1.0f;
  // side, uplo, trans, diag:
  // These parameters control the operation's behavior. For instance, side tells
  // cuBLAS whether the matrix A is on the left or right of B in the multiplication,
  // uplo indicates whether A is lower or upper triangular,
  // trans indicates if A should be transposed,
  // and diag specifies if A is unit triangular (diagonal elements are assumed to be 1).
  cublasStrmm(handle,
        CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER,
        CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
        B.size(0), 1,
        &alpha,
        B.data_ptr<float>(), B.size(1),
        v.data_ptr<float>(), v.size(0),
        Bv.data_ptr<float>(), Bv.size(0));

  at::Tensor ABv = at::empty({A.size(0)}, A.options());
  cublasStrmm(handle,
        CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER,
        CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
        A.size(0), 1,
        &alpha,
        A.data_ptr<float>(), A.size(1),
        Bv.data_ptr<float>(), Bv.size(0),
        ABv.data_ptr<float>(), ABv.size(0));

  // Destroy cuBLAS context
  cublasDestroy(handle);

  return ABv;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("multiply_low_rank_matrix", &multiply_low_rank_matrix, "Multiply low rank matrix (CUDA)");
}
