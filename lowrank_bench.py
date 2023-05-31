import torch
import torch.utils.benchmark as benchmark

n, m, k = 5000, 4000, 1000
A = torch.randn(n, k, device='cuda')
B = torch.randn(k, m, device='cuda')

A1 = torch.randn(n-k, k, device='cuda')
LU = torch.randn(k, k, device='cuda')
A2 = torch.randn(k, m-k, device='cuda')

v = torch.randn(m, device='cuda')

t0 = benchmark.Timer(
    stmt='lowrank.multiply_low_rank_matrix(A, B, v)',
    setup='''
        from torch.utils.cpp_extension import load
        lowrank = load('lowrank', ['lowrank.cpp', 'lowrank.cu'])
        ''',
    globals=dict(A=A, B=B, v=v))

t1 = benchmark.Timer(
    stmt='A @ (B @ v)',
    globals=dict(A=A, B=B, v=v))

t2 = benchmark.Timer(
    stmt='lowrank.multiply_low_rank_matrix(A, B, v)',
    setup='''
        from torch.utils.cpp_extension import load
        lowrank = load('lowrank', ['aten.cpp'])
        ''',
    globals=dict(A=A, B=B, v=v))

t3 = benchmark.Timer(
    stmt='lowrank.multiply_low_rank_matrix(A1, LU, A2, v)',
    setup='''
        from torch.utils.cpp_extension import load
        lowrank = load('lowrank', ['cublas3.cpp'])
        ''',
    globals=dict(A1=A1, LU=LU, A2=A2, v=v))

t4 = benchmark.Timer(
    stmt='lowrank.multiply_low_rank_matrix(A, B, v)',
    setup='''
        from torch.utils.cpp_extension import load
        lowrank = load('lowrank', ['cublas2.cpp'])
        ''',
    globals=dict(A=A, B=B, v=v))

print(t0.timeit(100))
print(t1.timeit(100))
print(t2.timeit(100))
print(t3.timeit(100))
print(t4.timeit(100))
