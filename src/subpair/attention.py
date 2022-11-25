from numba import cuda, float32
import numpy as np
import math
from cosine import outer, pairwise_cosine_cuda, pairwise_cosine
from time import monotonic

np.set_printoptions(formatter={"all": lambda x: str(x)})

# N, K = 4096 * 2 + 1, 512
N, K = 8192 + 1, 512

X = np.random.rand(N, K).astype(np.float32)
_ = outer(X)

start = monotonic()
Z = pairwise_cosine_cuda(X)
end = monotonic()
print(end - start)

start = monotonic()
control = pairwise_cosine(X)
end = monotonic()
print(end - start)

# print(Z)
# print(control)
assert np.allclose(Z, control, atol=1e-4)

# assert np.allclose(Z, control)
