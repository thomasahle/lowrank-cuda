#include <torch/extension.h>
#include <ATen/ATen.h>


void modified_gram_schmidt_inplace(at::Tensor& X) {
   // Orthogonalize and normalize the rows of X
   size_t n = X.size(0);
   for (size_t i = 0; i < n; ++i) {
      X.select(0, i) /= X.select(0, i).norm().item<float>();
      // Calculate the projection of all future vectors onto X[i]
      if (i + 1 != n) {
          // Calculate the projections of all future vectors onto the current vector
          auto proj = X.slice(0, i + 1, n).mv(X.select(0, i));
          // Subtract the projection from the future vectors
          // X.slice(0, i + 1, n) -= proj.unsqueeze(1) * X.select(0, i); but in-place
          X.slice(0, i + 1, n).addr_(proj, X.select(0, i), -1.0);
      }
   }
}

void gram_schmidt_inplace(at::Tensor& X) {
   // Orthogonalize and normalize the rows of X
   for (size_t i = 0; i < X.size(0); ++i) {
      if (i != 0) {
         // Calculate the projection of the current vector onto all previous vectors
         auto proj = X.slice(0, 0, i).mv(X.select(0, i));
         // Subtract the projection from the current vector
         // X.select(0, i) -= X.slice(0, 0, i).t().mv(proj); but in-place
         X.select(0, i).addmv_(X.slice(0, 0, i).t(), proj, -1.0);
      }
      X.select(0, i) /= X.select(0, i).norm().item<float>();
   }
}



at::Tensor subspace_iteration(at::Tensor A, int k, int num_iterations) {
   // What are we actually trying to do?
   // I guess find a low-rank approximation to A...
   // In the context of Favour, the idea is that we have a covariance matrix, Sigma ~ vv^T+i.
   // We now multiply it with a weight matrix, so (vv^T + i) W^T = v v^T W^T + i W^T
   // Ok, so we actually assume A to be square.

   //at::Tensor Qt = at::randn({k, A.size(1)}, A.options());
   // Alternative initialization
   //auto indices = torch::randperm(A.size(0), A.options().dtype(at::kLong)).slice(0, 0, k);
   //at::Tensor Qt = A.index_select(0, indices);
   // The lazyest initialization
   at::Tensor Qt = A.slice(0, 0, k).clone();

   // Gram-Schmidt process. For some reason I see no difference in accuracy between
   // the normal and modified process.
   //modified_gram_schmidt_inplace(Qt);
   gram_schmidt_inplace(Qt);

   for (int i = 0; i < num_iterations; ++i) {
      // Qt = (A Q).T = Q.T A
      // If A is row-major, doing Qt @ A.T is exactly what we want for performance.
      Qt = Qt.matmul(A.t());
      // We work with Q transposed since our gram-schmidt works row-wise.
      // Again, this is what we want as Qt is stored in row-major format.
      if (i % 2 == 0 || i + 1 == num_iterations)
         gram_schmidt_inplace(Qt);
   }

   return Qt;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> svd_right(at::Tensor A, int k, int num_iterations=100) {
   // If A is rectangular we can run the subspace iteration on whichever
   // of A^TA and AA^T are smaller. But generally I assume A to be square.
   at::Tensor AtA = A.t().mm(A);
   auto Vt = subspace_iteration(AtA, k, num_iterations);
   auto AV = A.mm(Vt.t());
   auto Rt = AV.t().mm(AV).diag();
   auto Sigma = at::sqrt(Rt);
   auto U = AV / Sigma;
   return {U, Sigma.diag(), Vt};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> svd_right2(at::Tensor A, int k, int num_iterations=100) {
   // If A is rectangular we can run the subspace iteration on whichever
   // of A^TA and AA^T are smaller. But generally I assume A to be square.
   at::Tensor AtA = A.t().mm(A);
   auto Vt = subspace_iteration(AtA, k, num_iterations);
   auto AVt = Vt.mm(A.t());
   auto Rt = AVt.mm(AVt.t()).diag();
   auto Sigma = at::sqrt(Rt);
   auto U = AVt.t() / Sigma;
   return {U, Sigma.diag(), Vt};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> svd_left(at::Tensor A, int k, int num_iterations=100) {
   at::Tensor AAt = A.mm(A.t());
   auto Ut = subspace_iteration(AAt, k, num_iterations);
   auto SVt = Ut.mm(A); // UtA = Ut U S Vt = SVt
   auto S2 = SVt.mm(SVt.t()).diag(); // S Vt (S Vt)^T = S Vt V S = S^2
   auto S = at::sqrt(S2);
   auto Vt = SVt / S.unsqueeze(1);
   return {Ut.t(), S.diag(), Vt};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> svd_left2(at::Tensor A, int k, int num_iterations=100) {
   at::Tensor AAt = A.mm(A.t());
   auto Ut = subspace_iteration(AAt, k, num_iterations);
   auto VS = A.t().mm(Ut.t());
   auto S2 = VS.t().mm(VS).diag();
   auto S = at::sqrt(S2);
   auto V = (VS / S);
   return {Ut.t(), S.diag(), V.t()};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
   m.def("qr", &gram_schmidt_inplace, "QR (ATen)");
   m.def("subspace_iteration", &subspace_iteration, "Subspace iteration (ATen)");
   m.def("svd_left", &svd_left, "SVD (ATen)");
   m.def("svd_left2", &svd_left2, "SVD (ATen)");
   m.def("svd_right", &svd_right, "SVD (ATen)");
   m.def("svd_right2", &svd_right2, "SVD (ATen)");
   //m.def("svd", &svd, "SVD (ATen)");
}
