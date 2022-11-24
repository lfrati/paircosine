from paircosine import *

import numpy as np
from scipy.spatial.distance import cosine
from itertools import combinations
from random import choice

import pytest

ATOL = 1e-6
N = 8
M = 6
K = 10


def scipy_cosine(v1, v2):
    N, M = v1.shape[0], v2.shape[0]
    ret = np.zeros((N, M))
    for i in range(N):
        for j in range(M):
            ret[i, j] = cosine(v1[i], v2[j])
    return ret


def get_test_data():
    feats1 = np.random.rand(N, K).astype(np.float32)
    feats2 = np.random.rand(M, K).astype(np.float32)
    target = scipy_cosine(feats1, feats2)
    self_target = scipy_cosine(feats1, feats1)
    return {
        "feats1": feats1,
        "feats2": feats2,
        "target": target,
        "self_target": self_target,
    }


@pytest.fixture(scope="session")
def data():
    return get_test_data()


# cosine_numba(feats1[0], feats1[1])
# self_paircosine_numba(feats1)
# self_paircosine_parallel(feats1)
# paircosine_numba(feats1, feats2)
# paircosine_parallel(feats1, feats2)


def test_cosine_distance():
    feats = np.random.rand(2, K).astype(np.float32)
    print(feats[0], feats[0].shape)
    res = cosine_distance(feats[0], feats[1])
    expected = cosine(feats[0], feats[1])
    assert np.allclose(res, expected, atol=ATOL)


def test_self_paircosine_numba(data):
    res = self_paircosine_numba(data["feats1"])
    assert np.allclose(res, data["self_target"], atol=ATOL)


def test_self_paircosine_parallel(data):
    res = self_paircosine_parallel(data["feats1"])
    assert np.allclose(res, data["self_target"], atol=ATOL)


def test_paircosine_numba(data):
    res = paircosine_numba(data["feats1"], data["feats2"])
    assert np.allclose(res, data["target"], atol=ATOL)


def test_paircosine_parallel(data):
    res = paircosine_numba(data["feats1"], data["feats2"])
    assert np.allclose(res, data["target"], atol=ATOL)


def test_paircosine_cuda(data):
    res = paircosine_cuda(data["feats1"], data["feats2"])
    assert np.allclose(res, data["target"], atol=ATOL)


def test_self_paircosine_cuda(data):
    res = self_paircosine_cuda(data["feats1"])
    assert np.allclose(res, data["self_target"], atol=ATOL)
