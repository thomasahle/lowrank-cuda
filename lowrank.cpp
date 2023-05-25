#include <torch/extension.h>
#include <ATen/ATen.h>

// CUDA forward declaration
void multiply_low_rank_matrix_cuda(
   torch::Tensor A,
   torch::Tensor B,
   torch::Tensor v,
   torch::Tensor r);

// Host function that wraps CUDA kernel
torch::Tensor multiply_low_rank_matrix(torch::Tensor A, torch::Tensor B, torch::Tensor v) {
  TORCH_CHECK(A.device().is_cuda(), "Tensor A must be a CUDA tensor");
  TORCH_CHECK(B.device().is_cuda(), "Tensor B must be a CUDA tensor");
  TORCH_CHECK(v.device().is_cuda(), "Tensor v must be a CUDA tensor");
  TORCH_CHECK(A.dim() == 2, "Tensor A must be a 2D tensor");
  TORCH_CHECK(B.dim() == 2, "Tensor B must be a 2D tensor");
  TORCH_CHECK(v.dim() == 1, "Tensor v must be a 1D tensor");
  TORCH_CHECK(A.size(1) == B.size(0), "The number of columns in A must match the number of rows in B");
  TORCH_CHECK(B.size(1) == v.size(0), "The number of columns in B must match the size of v");

  auto result = torch::zeros({A.size(0)}, A.options());
  multiply_low_rank_matrix_cuda(A, B, v, result);
  return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("multiply_low_rank_matrix", &multiply_low_rank_matrix, "Multiply low rank matrix (CUDA)");
}

