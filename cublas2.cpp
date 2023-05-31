#include <torch/extension.h>
#include <cublas_v2.h>

torch::Tensor multiply_low_rank_matrix(torch::Tensor A, torch::Tensor B, torch::Tensor v) {
  // Preliminary tensor checks
  // TORCH_CHECK(A.device().is_cuda(), "Tensor A must be a CUDA tensor");
  // TORCH_CHECK(B.device().is_cuda(), "Tensor B must be a CUDA tensor");
  // TORCH_CHECK(v.device().is_cuda(), "Tensor v must be a CUDA tensor");
  // TORCH_CHECK(A.dim() == 2, "Tensor A must be a 2D tensor");
  // TORCH_CHECK(B.dim() == 2, "Tensor B must be a 2D tensor");
  // TORCH_CHECK(v.dim() == 1, "Tensor v must be a 1D tensor");
  // TORCH_CHECK(A.size(1) == B.size(0), "The number of columns in A must match the number of rows in B");
  // TORCH_CHECK(B.size(1) == v.size(0), "The number of columns in B must match the size of v");

  // Create a handle for CUBLAS
  cublasHandle_t handle;
  cublasCreate(&handle);
  //cudaSetDevice(A.device().index());

  const float alpha = 1.0f;
  const float beta = 0.0f;

  // Compute the intermediate vector: Bv
  auto intermediate_vector = at::empty({B.size(0)}, B.options());
  cublasSgemv(handle, CUBLAS_OP_N, B.size(0), B.size(1), &alpha,
              B.data_ptr<float>(), B.size(0), v.data_ptr<float>(), 1, &beta, intermediate_vector.data_ptr<float>(), 1);

  // Compute the final result: Av
  auto result = torch::zeros({A.size(0)}, A.options());
  cublasSgemv(handle, CUBLAS_OP_N, A.size(0), A.size(1), &alpha,
              A.data_ptr<float>(), A.size(0), intermediate_vector.data_ptr<float>(), 1, &beta, result.data_ptr<float>(), 1);

  // Destroy the handle
  cublasDestroy(handle);

  return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("multiply_low_rank_matrix", &multiply_low_rank_matrix, "Multiply low rank matrix (CUDA)");
}
