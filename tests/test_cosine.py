from subpair import *

import numpy as np
from scipy.spatial.distance import cosine

import pytest

ATOL = 1e-6
N = 65
K = 20


def scipy_cosine(v1, v2):
    N, M = v1.shape[0], v2.shape[0]
    ret = np.zeros((N, M))
    for i in range(N):
        for j in range(M):
            ret[i, j] = cosine(v1[i], v2[j])
    return ret


def get_test_data():
    X = np.random.rand(N, K).astype(np.float32)
    target = scipy_cosine(X, X)
    return {
        "X": X,
        "target": target,
    }


@pytest.fixture(scope="session")
def data():
    return get_test_data()


def test_cosine_distance():
    X = np.random.rand(K).astype(np.float32)
    res = cosine_distance(X, X)
    expected = cosine(X, X)
    assert np.allclose(res, expected, atol=ATOL)


def test_pairiwse_numpy(data):
    res = pairwise_cosine(data["X"], mode="numpy")
    assert np.allclose(res, data["target"], atol=ATOL)


def test_pairwise_cuda(data):
    res = pairwise_cosine(data["X"], mode="cuda")
    assert np.allclose(res, data["target"], atol=ATOL)
