import torch
from torch.utils.cpp_extension import load
import tqdm
import torch.utils.benchmark as benchmark

ssi = load('ssi', ['ssi.cpp'], verbose=True,
extra_cflags=['-O3'],
extra_cuda_cflags=['-O3'],
        )
print(dir(ssi))

# Create some data
n, d, k = 5, 2, 2
num_iterations=10
A0 = torch.randn(n, d, device='cuda')
#A0 = A0 @ A0.T

# At = A0.T
# ssi.qr(At)
# print(At)
# print(At @ At.T)
# import sys; sys.exit()

# A = A0 @ A0.T
# R, Q = ssi.subspace_iteration(A, k, num_iterations)
# print(f'{A0=}')
# print(f'{R=}')
# print(f'{Q=}')

#print(f'{A0=}')
#U, S, Vt = ssi.svd(A0.contiguous(), k, num_iterations)
#print(f'{U=}')
#print(f'{S=}')
#print(f'{Vt=}')
#print(f'{U @ S @ Vt=}')
#
#U1, S1, Vt1 = torch.linalg.svd(A0, full_matrices=False)
#S1 = torch.diag(S1)
#print(f'{U1=}')
#print(f'{S1=}')
#print(f'{Vt1=}')
#print(f'{U1 @ S1 @ Vt1=}')
#
#torch.testing.assert_close(U @ S @ Vt, A0)
#torch.testing.assert_close(U1 @ S1 @ Vt1, A0)

if False:
    n, d, k = 1000, 100, 10
    A0 = torch.randn(n, d, device='cuda')
    A0 = (A0 @ A0.T).contiguous()
    for _ in range(2):
        for func in ["svd_left", "svd_left2", "svd_right", "svd_right2"]:
            t0 = benchmark.Timer(
                stmt=f'ssi.{func}(A0, k, num_iterations)',
                globals=dict(ssi=ssi, A0=A0, k=10, num_iterations=10))
            print(t0.timeit(100))

if True:
    for _ in tqdm.tqdm(range(10)):
        n, d = 100, 10
        A0 = torch.randn(n, d, device='cuda')
        A0 = A0 @ A0.T
        U, S, Vt = ssi.svd_left(A0.contiguous(), d, num_iterations)
        E = U @ S @ Vt
        print((E-A0).norm())
        torch.testing.assert_close(E, A0, atol=.1, rtol=.1)
    print("Success!")
