import math
import numpy as np
from numba import cuda, njit, prange

from typing import Callable

def cosine_distance(u: np.ndarray, v: np.ndarray):
    uv = 0
    uu = 0
    vv = 0
    for i in range(u.shape[0]):
        uv += u[i] * v[i]
        uu += u[i] * u[i]
        vv += v[i] * v[i]
    cos_theta = 1
    if uu != 0 and vv != 0:
        cos_theta = 1 - uv / math.sqrt(uu * vv)
    return cos_theta


# =============================================================================
#                                CPU
# =============================================================================


cosine_numba = njit("f4(f4[:],f4[:])", fastmath=True, cache=True)(cosine_distance)


def self_paircosine_cpu(features):
    """
    Compute pairwise cosine distances in parallel, using numba.
    features  : (N,K) features matrix
    distances : (N,N) distances matrix
    """
    N = features.shape[0]
    distances = np.zeros((N, N), dtype=np.float32)
    for row in prange(N):
        for other in range(row + 1):
            d = cosine_numba(features[row], features[other])
            distances[row, other] = d
            distances[other, row] = d
    return distances


def paircosine_cpu(feats1, feats2):
    """
    Compute pairwise cosine distances in parallel, using numba.
    features  : (N,K) features matrix
    distances : (N,N) distances matrix
    """
    N = feats1.shape[0]
    M = feats2.shape[0]
    distances = np.zeros((N, M), dtype=np.float32)
    for row in prange(N):
        for other in range(M):
            d = cosine_numba(feats1[row], feats2[other])
            distances[row, other] = d
    return distances


self_paircosine_numba = njit("f4[:,:](f4[:,:])", parallel=True, cache=True)(
    self_paircosine_cpu
)
self_paircosine_parallel = njit("f4[:,:](f4[:,:])", parallel=False, cache=True)(
    self_paircosine_cpu
)

paircosine_numba = njit("f4[:,:](f4[:,:], f4[:,:])", parallel=True, cache=True)(
    paircosine_cpu
)
paircosine_parallel = njit("f4[:,:](f4[:,:], f4[:,:])", parallel=False, cache=True)(
    paircosine_cpu
)

# =============================================================================
#                                GPU
# =============================================================================

cosine_cuda = cuda.jit(device=True)(cosine_distance)

# set fastmath for the sqrt in the device function
@cuda.jit(fastmath=True)
def self_cosine_kernel(features, distances):
    """
    Compute one entry of the distances matrix
    features  : (N,K) features matrix
    distances : (N,N) distances matrix
    """
    i, j = cuda.grid(2)
    if i < features.shape[0] and j <= i:
        sim = cosine_cuda(features[i], features[j])
        distances[i, j] = sim
        distances[j, i] = sim


def self_paircosine_cuda(features):
    """
    Compute pairwise cosine distances with cuda acceleration.
    features  : (N,K) features matrix
    distances : (N,N) distances matrix
    """
    features = features.astype(np.float32)
    N = features.shape[0]
    distances = np.zeros((N, N), dtype=np.float32)
    features_cuda = cuda.to_device(features)
    distances_cuda = cuda.to_device(distances)

    threadsperblock = (16, 16)
    blockspergrid_x = math.ceil(N / threadsperblock[0])
    blockspergrid_y = math.ceil(N / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    self_cosine_kernel[blockspergrid, threadsperblock](features_cuda, distances_cuda)
    distances = distances_cuda.copy_to_host()
    return distances


# set fastmath for the sqrt in the device function
@cuda.jit(fastmath=True)
def cosine_kernel(v1, v2, distances):
    """
    Compute one entry of the distances matrix
    v1        : (N,K) features matrix
    v2        : (M,K) features matrix
    distances : (N,M) distances matrix
    """
    i, j = cuda.grid(2)
    if i < v1.shape[0] and j < v2.shape[0]:
        sim = cosine_cuda(v1[i], v2[j])
        distances[i, j] = sim


def paircosine_cuda(v1, v2=None):
    """
    Compute pairwise cosine distances with cuda acceleration.
    v1        : (N,K) features matrix
    v2        : (M,K) features matrix
    distances : (N,M) distances matrix
    """
    if v2 is None:
        return paircosine_NN_cuda(v1)
    v1 = v1.astype(np.float32)
    v2 = v2.astype(np.float32)
    distances = np.zeros((v1.shape[0], v2.shape[0]), dtype=np.float32)
    v1_cuda = cuda.to_device(v1)
    v2_cuda = cuda.to_device(v2)
    distances_cuda = cuda.to_device(distances)

    threadsperblock = (16, 16)
    blockspergrid_x = math.ceil(v1.shape[0] / threadsperblock[0])
    blockspergrid_y = math.ceil(v2.shape[0] / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    cosine_kernel[blockspergrid, threadsperblock](v1_cuda, v2_cuda, distances_cuda)
    distances = distances_cuda.copy_to_host()
    return distances
