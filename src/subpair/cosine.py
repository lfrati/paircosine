import math
from numba import cuda, njit, float32
import numpy as np

# =============================================================================
#                       Reference Implemenation
# =============================================================================


@njit("f4(f4[:], f4[:])")
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


def pairwise_cosine_numpy(X):
    norms = np.einsum("ij,ij->i", X, X)
    np.sqrt(norms, norms)
    X /= norms[:, np.newaxis]
    dists = 1 - np.dot(X, X.T)
    return dists


# =============================================================================
#                                GPU
# =============================================================================


# Controls threads per block and shared memory usage.
# The computation will be done on blocks of TPBxTPB elements.
# TPB should not be larger than 32 in this example
TPB = 16


@cuda.jit
def outer_kernel(A, C):
    """
    Perform matrix multiplication of C = A * A.T using CUDA shared memory.

    Reference: https://stackoverflow.com/a/64198479/13697228 by @RobertCrovella
    """
    # Define an array in the shared memory
    # The size and type of the arrays must be known at compile time
    sA = cuda.shared.array(shape=(TPB, TPB), dtype=float32)
    sB = cuda.shared.array(shape=(TPB, TPB), dtype=float32)

    x, y = cuda.grid(2)

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bpg = cuda.gridDim.x  # blocks per grid

    # Each thread comptes one element in the result matrix.
    # The dot product is chunked into dot products of TPB-long vectors.
    tmp = float32(0.0)
    for i in range(bpg):
        # Preload data into shared memory
        sA[ty, tx] = 0
        sB[ty, tx] = 0
        if y < A.shape[0] and (tx + i * TPB) < A.shape[1]:
            sA[ty, tx] = A[y, tx + i * TPB]
        if x < A.shape[0] and (ty + i * TPB) < A.shape[1]:
            sB[ty, tx] = A[x, ty + i * TPB]

        # Wait until all threads finish preloading
        cuda.syncthreads()

        # Computes partial product on the shared memory
        for j in range(TPB):
            tmp += sA[ty, j] * sB[j, tx]

        # Wait until all threads finish computing
        cuda.syncthreads()
    if y < C.shape[0] and x < C.shape[1]:
        C[y, x] = tmp
        # C[x, y] = tmp


def outer(X):
    x_d = cuda.to_device(X)
    z_h = np.empty([X.shape[0], X.shape[0]], dtype=np.float32)
    z_d = cuda.to_device(z_h)

    threadsperblock = (TPB, TPB)
    blockspergrid_x = math.ceil(z_h.shape[0] / threadsperblock[0])
    blockspergrid_y = math.ceil(z_h.shape[1] / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    outer_kernel[blockspergrid, threadsperblock](x_d, z_d)
    z_h = z_d.copy_to_host()
    return z_h


def pairwise_cosine_cuda(X):
    norms = np.einsum("ij,ij->i", X, X)
    np.sqrt(norms, norms)
    X /= norms[:, np.newaxis]
    dists = 1 - outer(X)
    return dists


def pairwise_cosine(X, mode="numpy"):
    if mode == "numpy":
        return pairwise_cosine_numpy(X)
    return pairwise_cosine(X)


__all__ = ["pairwise_cosine", "cosine_distance"]
