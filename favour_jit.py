import torch
from torch.utils.cpp_extension import load
import tqdm
import torch.utils.benchmark as benchmark
import time

ssi = load('ssi', ['ssi.cpp'], verbose=True,
extra_cflags=['-O3'],
extra_cuda_cflags=['-O3'],
        )
print(dir(ssi))

torch.manual_seed(42)

# Create some data
d, k = 256, 10
num_iterations=10
opts = dict(device='cuda')
Ut = torch.randn(k, d, **opts)
#D = torch.randn(d, **opts) ** 2
D = torch.randn(d, **opts).clamp_min(0)
W = torch.randn(d, d, **opts)

M0 = Ut.T @ Ut + D.diag()
M1 = W @ M0 @ W.T

WU = W @ Ut.T
WUUtWt = WU @ WU.T
WDWt = (W * D) @ W.T
M1s = WUUtWt + WDWt
torch.testing.assert_close(M1, M1s, atol=1e-3, rtol=1e-3)

# print(f'{M1=}')

#U1, S1, Vt1 = ssi.svd_left(M1, k, num_iterations)
#U1, S1, Vt1 = torch.svd(M1, k, num_iterations)
U1, S1, V1 = torch.svd_lowrank(M1, k, num_iterations)
Vt1, S1 = V1.T, S1.diag()
torch.testing.assert_close(U1.T, Vt1, rtol=1e-3, atol=1e-3)
# print(f'{Vt1=}')
M2 = U1 @ S1 @ Vt1
# print(f'{M2=}')
print("m2 error", (M1-M2).norm())

start = time.time()
for niter in [1, 2, 5, 10, 20, 50, 100, 200, 500]:
    D2, Ut2 = ssi.fast_dplr(Ut, D, W, niter)
    M3 = Ut2.T @ Ut2 + D2.diag()
    print(niter, (M1-M3).norm())
#print(f'{M3=}')
print('took', time.time() - start)
#print(D2)
