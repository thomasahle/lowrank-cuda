#include <torch/extension.h>
#include <ATen/ATen.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

// A = [X ] (n-k x k)
//     [LU] (kxk)
//     [Y ] (m-k x k)

// [L ]
// [A1] [U A2] B

// [U A2] B = U B + A2 B

// Maybe if we moved everything to the right side, this would work better?
// Say we computed
// B [U \\ A2] [L & A1]
// Then B [U \\ A2] = B[:,:k] U + B[:, k:] A2, meaning we split by columns.
// And I [L & A1] = [I L & I A1], meaning we concat by columns.

//                 B.shape = (l, m)
// So now we have A1.shape = (k, n-k)
// and            A2.shape = (m-k, k)
//                 I.shape = (l, k)

at::Tensor multiply_low_rank_matrix(at::Tensor A1, at::Tensor LU, at::Tensor A2, at::Tensor B) {
    cudaSetDevice(A1.device().index());

    cublasHandle_t handle;
    cublasCreate(&handle);

    const float one = 1.0f;
    const float zero = 0.0f;

    // Dimensions
    int k = LU.size(0);
    TORCH_CHECK(k == LU.size(1), "LU must be square");
    int n = A1.size(1) + k;
    TORCH_CHECK(k == A1.size(0), "A1 must have k rows");
    TORCH_CHECK(k == A2.size(1), "A2 must have k cols");
    int m = A2.size(0) + k;
    int l = B.size(0);
    TORCH_CHECK(m == B.size(1), "B must have m cols");

    // Intermediate tensor
    auto I = at::empty({l, k}, A1.options());

    // Compute I = B[:, :k] U
    cublasStatus_t status = cublasStrmm(handle,
          // Multiply on left side with upper triangular matrix
          CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER,
          // Don't transpose it, and don't use a unit triangular (diagonal = 1s)
          CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT,
          l, k, // Dimension of B[:, :k]
          &one,
          LU.data_ptr<float>(), LU.size(0),
          B.data_ptr<float>(), B.size(0),
          I.data_ptr<float>(), I.size(0));
    TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS, "B U didn't succeed");

    // Ok, this is the problem:
    // cublas assumes column-major format, but our matrices and stuff are in
    // row-major. This means, it interprets
    // [1, 1, 1] [1 1 1] = [1 2 3]
    // [1, 1, 1] [0 1 1] = [1 2 3]
    //           [0 0 1]
    // As
    // [1 1]
    // [2 2]
    // [3 3]
    // and then writes that to the output, which we then reinterpret as row-major:
    // [1 1 2]
    // [2 3 3].
    // I have to really try to embrace the whole transposed thing to get this right.

    // This https://stackoverflow.com/questions/56043539/cublassgemm-row-major-multiplication
    // say to "slyly" try to compute
    // C^T instead of C. Then when cublas dumps it into our array, we interpret it in row-major,
    // and end up getting what we actually wante, C.

    // Say C = B A^T, then C^T = A B^T.
    // We have B in row-major format, but the way cublas reads it, it actually looks like B^T.
    // This is also good, because in matrix multiplication we need the columns of the right argument
    // to be together, so which fits with how we stored B.
    // Now, we also need A. If we pass A to cublas, it will look like A.T, so we use the OP_T operation
    // to make it multiply with the transpose, giving us A^T^T = A.
    // This works well because we need the rows of the left side of the mm to have rows together,
    // which they will now.

    // Compute C2 = B[:, k:] A2, add it to C1 to get the final result in I
    /*
    status = cublasSgemm(handle,
          CUBLAS_OP_N, CUBLAS_OP_N,
          l, k, m - k, // B.rows, A2.cols, B[:,:k].cols
          &one,
          // Skip k colums, each of m vectors
          B.data_ptr<float>() + k*l, B.size(0),
          A2.data_ptr<float>(), A2.size(0),
          &one, // Add the result to I, keeping the original value
          I.data_ptr<float>(), I.size(0));
    TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS, "B A2 didn't succeed");
    */

    // Prepare the output tensor and get a raw pointer to its data
    auto C = at::empty({l, n}, A1.options());

    // Write the output of I L to the beginning of C
    status = cublasStrmm(handle,
          // Multiply on the left side with the lower triangular matrix
          CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER,
          // Don't transpose it, but use a unit triangular (diagonal = 1s)
          CUBLAS_OP_N, CUBLAS_DIAG_UNIT,
          I.size(0), I.size(1),
          &one,
          LU.data_ptr<float>(), LU.size(0),
          I.data_ptr<float>(), I.size(0),
          C.data_ptr<float>(), C.size(0));
    TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS, "I L didn't succeed");

    // Write the output of I A1 to the end of C
    status = cublasSgemm(handle,
          CUBLAS_OP_N, CUBLAS_OP_N,
          l, n-k, k, // I.rows, A1.cols, I.cols
          &one,
          I.data_ptr<float>(), I.size(0),
          A1.data_ptr<float>(), A1.size(0),
          &one,
          // Offset C by k cols of size l
          C.data_ptr<float>() + k*l, C.size(0));
    TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS, "I A1 didn't succeed");

    cublasDestroy(handle);
    return I;
    //return C;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("multiply_low_rank_matrix", &multiply_low_rank_matrix, "Multiply low rank matrix (CUDA)");
}
