import numpy as np
from src.subpair.cosine import *
from time import monotonic
from numba import cuda

from sklearn.metrics.pairwise import cosine_distances


CNT = 4


def timeit(f):
    times = []
    X = np.random.rand(N, K).astype(np.float32)
    _ = f(X)
    avg = 0.0
    for _ in range(CNT):
        X = np.random.rand(N, K).astype(np.float32)
        start = monotonic()
        _ = f(X)
        end = monotonic()
        times.append(end - start)
    avg = np.mean(times)
    return avg


N, K = 256, 2304
avg = 0.0

while avg < 1.0:
    N = N * 2 + 1

    GOPs = N * N * K / 1e9

    print(f"{N=} {K=} {GOPs=:.0f}")

    avg1 = timeit(cosine_distances)
    print(f"  sklearn: {avg1:.2f}s - {GOPs / avg1:.1f} GFLOPS")

    avg2 = timeit(pairwise_cosine)
    print(f"    numpy: {avg2:.2f}s - {GOPs / avg2:.1f} GFLOPS")

    if cuda.is_available():
        print(cuda.detect())
        avg3 = timeit(pairwise_cosine_cuda)
        print(f"     cuda: {avg3:.2f}s - {GOPs / avg3:.1f} GFLOPS")
    else:
        avg3 = 0.0
    print()

    avg = max([avg1, avg, avg3])
