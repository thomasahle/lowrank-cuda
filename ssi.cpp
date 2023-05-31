#include <torch/extension.h>
#include <ATen/ATen.h>


void modified_gram_schmidt_inplace(at::Tensor& X) {
   // Orthogonalize and normalize the rows of X
   int64_t n = X.size(0);
   at::Tensor proj = at::empty({n}, X.options());
   for (int64_t i = 0; i < n; ++i) {
      auto v = X.select(0, i);
      // We could decide to not normalize v here, and divide everything
      // in the next bit by the norm instead.
      v /= v.norm().item<float>();
      // Calculate the projection of all future vectors onto X[i]
      if (i + 1 != n) {
          // Calculate the projections of all future vectors onto the current vector
          // auto proj = X.slice(0, i + 1, n).mv(v);
          // See if we can reuse the memory
          proj.resize_({n - i - 1});
          at::mv_out(proj, X.slice(0, i + 1, n), v);
          // Subtract the projection from the future vectors
          // X.slice(0, i + 1, n) -= proj.unsqueeze(1) * v; but in-place
          X.slice(0, i + 1, n).addr_(proj, v, -1.0);
      }
   }
}

void unnormalized_modified_gram_schmidt_inplace(at::Tensor& X) {
   // Orthogonalize and normalize the rows of X
   int64_t n = X.size(0);
   at::Tensor proj = at::empty({n}, X.options());
   for (int64_t i = 0; i < n; ++i) {
      auto v = X.select(0, i);
      // We could decide to not normalize v here, and divide everything
      // in the next bit by the norm instead.
      auto norm2 = (v * v).sum();
      // Calculate the projection of all future vectors onto X[i]
      if (i + 1 != n) {
          // Calculate the projections of all future vectors onto the current vector
          // auto proj = X.slice(0, i + 1, n).mv(v);
          // Trying to reuse the memory...
          proj.resize_({n - i - 1});
          at::mv_out(proj, X.slice(0, i + 1, n), v);
          proj /= norm2;
          // Subtract the projection from the future vectors
          // X.slice(0, i + 1, n) -= proj.unsqueeze(1) * v; but in-place
          X.slice(0, i + 1, n).addr_(proj, v, -1.0);
      }
   }
}


void gram_schmidt_inplace(at::Tensor& X) {
   // Orthogonalize and normalize the rows of X
   for (size_t i = 0; i < X.size(0); ++i) {
      auto v = X.select(0, i);
      if (i != 0) {
         // Calculate the projection of the current vector onto all previous vectors
         auto proj = X.slice(0, 0, i).mv(v);
         // Subtract the projection from the current vector
         // v -= X.slice(0, 0, i).t().mv(proj); but in-place
         v.addmv_(X.slice(0, 0, i).t(), proj, -1.0);
      }
      v /= v.norm().item<float>();
   }
}

std::tuple<at::Tensor, at::Tensor> fast_dplr(at::Tensor Ut, at::Tensor diag, at::Tensor W, int niter) {
   // Computes the lowrank + diag decomposition of W (UU^T + diag) W^T.
   int k = Ut.size(0);
   int d0 = Ut.size(1);
   int d1 = W.size(0);

   //std::cout << "A" << std::endl;

   //auto WDWt = (W * diag.unsqueeze(1)).mm(W.t());
   auto WDWt = (W * diag).mm(W.t());
   auto WU = W.mm(Ut.t());
   auto WUUtWt = WU.mm(WU.t());
   // Maybe we don't even want to compute the full M?
   auto M = WDWt + WUUtWt;
   auto Md = M.diag();

   //std::cout << "B" << std::endl;

   at::Tensor D = Md.clone();
   at::Tensor S;
   at::Tensor Vt = at::randn({k, d1}, Ut.options());

   // This is the version that actually makes sense.
   // It's just a bit slower than the others...
   at::Tensor VtMt = Vt.clone();
   if (false)
   for (int i = 0; i < niter; i++) {
      // If I don't do GS every iteration, my rows are not unit-norm, which I expect below
      //if (i % 2 == 0 || i + 1 == niter)
      // GS could return the norms, but with the current code ordering that wouldn't work,
      // since what we want is the norms after multiplying with M.
      //modified_gram_schmidt_inplace(Vt);
      // Somehow this unnormalized version is faster...
      unnormalized_modified_gram_schmidt_inplace(Vt);
      Vt /= Vt.norm(2, 1).unsqueeze(1);

      // Vt = Vt @ (M - D). The rest is commentary.
      //auto VtMt = Vt.mm((M - D.diag()).t());
      //VtMt.copy_(Vt.mm((M - D.diag()).t()));
      VtMt = Vt.mm((M - D.diag()).t());
      //at::mm_out(VtMt, Vt, (M - D.diag()).t());
      // We could maybe improve this by not even computing M outside the loop,
      // but dicretly computing Vt (WDWt + WUUtWt) here. Though it would only be
      // smart if we are doing few iterations (which we are).
      //at::mm_out(VtMt, Vt, M.t());  // VtMt = Vt.mm(M.t())
      // This is still allocating.
      //VtMt.sub_(Vt * D);  // VtMt -= Vt * D

      // Update the diagonal approximation
      // auto S2 = VtMt.mm(VtMt.t()).diag();  // (MV)^T MV = V^T (U S V^T)^T (U S V^T) V = S^2
      // S = at::sqrt(S2);
      S = VtMt.norm(2, 1); // column norms of Vt
      //assert(S.size(0) == k);
      // Might as well do this even in the last loop?
      //D = Md - (Vt.t() * S).mm(Vt).diag();
      D = Md - (Vt * Vt).t().mv(S);
      D.clamp_min_(0);
      if (i + 1 != niter) {
         // Update Vt in the normal supspace iteration way
         Vt = VtMt;
         //Vt.copy_(VtMt);
      }
   }

   // This is the version in the paper, which uses unormalized GS.
   // The advantage is that it only requires computing the norm once.
   if (false)
   for (int i = 0; i < niter; i++) {
      // The paper actually calls for "un-normalized gram schmidt"
      unnormalized_modified_gram_schmidt_inplace(Vt);
      // If we use normalized gram schmidt, it seems these will all just be one...
      S = Vt.norm(2, 1); // L2 norm of each row
      assert(S.size(0) == k);
      Vt /= S.unsqueeze(1);
      // There seems to be some double work going on here...
      D = Md - (Vt * Vt * S.unsqueeze(1)).sum(0);
      assert(D.size(0) == d1);
      D.clamp_min_(0);
      if (i + 1 != niter) {
         Vt = Vt.mm((M - D.diag()).t());
      }
   }
   // We put just half of the diagonal, S, in Vt, since the full matrix is VV^T
   return {D, Vt * at::sqrt(S).unsqueeze(1)};
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
   // FIXME: It's obviously inefficient to compute the whole AV(AV)^T product
   // if I only need the diagonal
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
   m.def("fast_dplr", &fast_dplr, "Fast DPLR Decomposition (ATen)");
   //m.def("svd", &svd, "SVD (ATen)");
}
