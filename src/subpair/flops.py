import numpy as np
from cosine import *
from time import monotonic

from sklearn.metrics.pairwise import cosine_distances

CNT = 4

N, K = 4096 * 2 + 1, 2304


def timeit(f):
    times = []
    X = np.random.rand(N, K).astype(np.float32)
    _ = f(X)
    for _ in range(CNT):
        X = np.random.rand(N, K).astype(np.float32)
        start = monotonic()
        _ = f(X)
        end = monotonic()
        times.append(end - start)
    avg = np.mean(times)
    return avg


#%%

# sk = cosine_distances(f1)
# cu = pairwise_cosine_cuda(f1)
# assert np.allclose(sk, cu)

GOPs = N * N * K / 1e9

# avg = timeit(cosine_distances)
# print(f"sklearn: {avg:.2f}s - {GOPs / avg:.1f} GFLOPS")

avg = timeit(pairwise_cosine)
print(f"  numpy: {avg:.2f}s - {GOPs / avg:.1f} GFLOPS")

avg = timeit(pairwise_cosine_cuda)
print(f"   cuda: {avg:.2f}s - {GOPs / avg:.1f} GFLOPS")

#%%
