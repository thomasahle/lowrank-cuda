#include <torch/extension.h>
#include <ATen/ATen.h>

at::Tensor multiply_low_rank_matrix(at::Tensor A, at::Tensor B, at::Tensor v) {
  // Ensure the input tensors are on the same device and have the correct dimensionality
  //TORCH_CHECK(A.device() == B.device() && B.device() == v.device(), "All tensors must be on the same device");
  //TORCH_CHECK(A.dim() == 2 && B.dim() == 2 && v.dim() == 1, "Tensors must have correct dimensions: A(2D), B(2D), v(1D)");

  // Calculate Bv
  at::Tensor intermediate_vector = at::matmul(B, v);
  
  // Calculate Av
  at::Tensor result = at::matmul(A, intermediate_vector);

  return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("multiply_low_rank_matrix", &multiply_low_rank_matrix, "Multiply low rank matrix (CUDA)");
}
