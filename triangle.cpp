#include <torch/extension.h>
#include <cuda_runtime.h>

// CUDA forward declarations
void triangle_multiply_cuda(
    torch::Tensor matrix,
    torch::Tensor vector,
    torch::Tensor result);

#define CHECK_CUDA(x) AT_ASSERTM(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) { CHECK_CUDA(x); CHECK_CONTIGUOUS(x); }

void triangle_multiply(torch::Tensor matrix, torch::Tensor vector, torch::Tensor result) {
  CHECK_INPUT(matrix);
  CHECK_INPUT(vector);
  CHECK_INPUT(result);
  triangle_multiply_cuda(matrix, vector, result);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("triangle_multiply_cuda", &triangle_multiply, "Multiply vector by triangular matrix (CUDA)");
}
