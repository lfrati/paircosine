from collections import defaultdict

import numpy as np
from numba import njit, prange
from numpy.random import choice
from tqdm import trange


@njit
def subset(matrix, cols):
    """
    Optimized version of np.ix_ for symmetric mat with zero diag

    N,C = 5,3 # N size of matrix, C columns selected
    m = np.random.rand(N, N)
    m = (m + m.T)/2 # symmetric
    np.fill_diagonal(m,0)
    select = np.sort(np.random.choice(np.arange(N), size=C, replace=False))
    sub = m[np.ix_(select, select)]
    triu = np.triu_indices(len(sub), k=1)
    assert np.allclose(sub[triu], subset(m, select))

    e.g. N = 5, C = 3
    m:
          | 0  a  b  c  d |
          | a  0  e  f  g |
          | b  e  0  h  i |
          | c  f  h  0  j |
          | d  g  i  j  0 |

    select: [1,3,4]

    sub:
          | 0  f  g |
          | f  0  j |
          | g  j  0 |

    sub[triu]: [f, g, j] # len = 3*(3-1)//2
    """
    C = cols.shape[0]
    res = np.empty((C * (C - 1)) // 2, dtype=matrix.dtype)
    k = 0
    for i in range(C):
        for j in cols[i + 1 :]:
            res[k] = matrix[cols[i], j]
            k += 1
    return res


@njit
def mutate(parent, valid, loci, M):
    """
    Args:
        parent (np.ndarray): Gene to be mutated, a list of S selected indices.
        valid (np.ndarray): List of indices within the whole matrix which can be chosen as replacements.
        loci (np.ndarray): List of indices within `parent` that can be replaced.
        M (int): Number of mutations.
    """
    # how to mutate (only new values)
    mutations = choice(valid, size=M, replace=False)
    # where to mutate
    positions = choice(loci, size=M, replace=False)
    child = np.copy(parent)
    child[positions] = mutations
    return child


@njit
def numba_ix(arr, rows, cols):
    """
    Numba compatible implementation of arr[np.ix_(rows, cols)] for 2D arrays.
    :param arr: 2D array to be indexed
    :param rows: Row indices
    :param cols: Column indices
    :return: 2D array with the given rows and columns of the input array
    """
    one_d_index = np.zeros(len(rows) * len(cols), dtype=np.int32)
    for i, r in enumerate(rows):
        start = i * len(cols)
        one_d_index[start: start + len(cols)] = cols + arr.shape[1] * r

    arr_1d = arr.reshape((arr.shape[0] * arr.shape[1], 1))
    slice_1d = np.take(arr_1d, one_d_index)
    return slice_1d.reshape((len(rows), len(cols)))


@njit
def mutate_deterministic(parent, valid, distances):
    """
    Args:
        parent (np.ndarray): Gene to be mutated, a list of S selected indices.
        valid (np.ndarray): List of indices within the whole matrix which can be chosen as replacements.
        distances (np.ndarray): distance matrix of shape [N,N]
    """
    distgrid = numba_ix(distances, parent, parent)  # distances[np.ix_(parent, parent)]
    to_replace = distgrid.sum(axis=0).argmax()
    to_keep = list(parent[:to_replace]) + list(parent[to_replace+1:])
    # Below, each row is an existing point and each column is a candidate new point to add to the set. Which candidate
    # point has the lowest total distance to all the existing points?
    candidates = numba_ix(distances, to_keep, valid)  # distances[np.ix_(to_keep, valid)]
    candidate_scores = candidates.sum(axis=0)
    best_replacement = candidate_scores.argmin()
    # proportional_choice = choice(valid, p=candidate_scores / candidate_scores.max())
    child = np.copy(parent)
    child[to_replace] = valid[best_replacement]
    return child


@njit
def cross(dad, mom):
    valid = np.array(list(set(dad) | set(mom)))  # cast to np for numba reasons
    child = choice(valid, size=len(dad), replace=False)
    return child


def score(pop, distances):
    fitnesses = np.array([subset(distances, m).sum() for m in pop])
    return fitnesses


@njit(parallel=True)
def step(pop, fitnesses, P, S, K, M, O, R, space, distances):
    """
    Given a population and precomputed fitnesses, returns the next generation after selection, mutation, and crossover.

    Args:
        pop (np.ndarray) : array of shape [P, S]
        fitnesses (np.ndarray) : array of shape [P], each value corresponding to a row of `pop`
        P (int): population size
        S (int): subset size (number of matrix columns to select)
        K (int): number of parents (P-K children)
        M (int): number of mutations per child
        O (int): fraction of crossovers e.g. O=2 -> 1/2, O=10 -> 1/10, (bigger=faster)
        R (float): probability of random mutation (as opposed to deterministic mutation)
        space (set): The set of possible column indices: range(0 .. # columns).
        distances (np.ndarray): distance matrix of shape [N,N]

    Returns:
       pop (np.ndarray) : the next generation, shape [P, S]
    """
    indexes = np.argsort(fitnesses)
    fit_idxs = indexes[:K]
    unfit_idxs = indexes[K:]
    top_k = pop[fit_idxs]
    loci = np.arange(S)
    # compute valid indices to sample from during mutation
    valid = np.empty((K, len(space) - S), dtype=pop.dtype)
    for i in prange(K):
        valid[i] = np.array(list(space - set(top_k[i])))
    children = np.empty(((P - K), S), dtype=pop.dtype)

    ##### MUTATION ######
    for i in prange(P - K):
        j = i % K
        if np.random.random() <= R:
            children[i] = mutate(top_k[j], valid[j], loci, M)
        else:
            children[i] = mutate_deterministic(top_k[j], valid[j], distances)

    ##### CROSSOVER ######
    for i in prange(0, P - K - 1):
        if i % O == 0:  # only cross 1/O of the offspring
            # children are arranged in sorted bands so fit[i]>fit[i+1] (apart from in between bands)
            children[i + 1] = cross(children[i], children[i + 1])

    pop[unfit_idxs] = children

    return pop


def extract(
    distances,
    P,
    S,
    K,
    M,
    O,
    R,
    its,
):
    """Given a matrix of pairwise distances evolves the minimal subset of size S.
    (minimal = sum of entries in the subset)

    Note:
        The input matrix is assumed to be symmetric but it's not checked. Beware.

    Args:
        distances (np.ndarray): distance matrix of shape [N,N]
        P (int): population size
        S (int): subset size (number of matrix columns to select)
        K (int): number of parents (P-K children)
        M (int): number of mutations per child
        O (int): fraction of crossovers e.g. O=2 -> 1/2, O=10 -> 1/10, (bigger=faster)
        R (float): probability of random mutation (as opposed to deterministic mutation)
        its (int): number of generations to run evolution

    Returns:
       best (np.ndarray) : S indeces corresponding to the minimal subset found
       stats (dict)      : stats["bests"] = best subset found at each iteration
                           stats["fits"] = best fitness found at each iteration
    """
    N = distances.shape[0]
    assert S <= N, f"Error: value of S={S} is > than the #distances={N}"
    assert O >= 1, f"Error: value of O={O} is < 1"
    assert P >= K, f"Error: value of P={P} is < K"
    space = set(range(N))
    pop = np.array([np.random.choice(list(space), size=S, replace=False) for _ in range(P)])
    fitnesses = np.zeros(P)

    stats = defaultdict(list)
    for _ in trange(its):

        pop = step(pop, fitnesses, P, S, K, M, O, R, space, distances)

        fitnesses = score(pop, distances)

        best = np.copy(pop[np.argmin(fitnesses)])
        stats["bests"].append(best)
        stats["fits"].append(fitnesses.min())

    fitnesses = score(pop, distances)
    best = np.copy(pop[np.argmin(fitnesses)])

    return best, stats


__all__ = ["extract"]
