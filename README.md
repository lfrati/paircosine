<p align="center">
    <img width="300" alt="Logo" src="https://user-images.githubusercontent.com/3115640/203899211-fff1c9d8-10cd-4a84-88b5-518a591cd1e5.jpeg">
    <p align="center">/sʌb.pɛɹ/</p>
</p>

# SubPair  ![CI](https://github.com/lfrati/subpair/actions/workflows/test.yml/badge.svg)

> "All you need is love and _evolutionary matrix subset extraction_." - J. Lennon

Pairwise cosine distance is great to easily compare many vectors. However, you can end up with a very sizeable distance matrix. What if you would like to find a small subset of that matrix? Let's search it by evolution.

Given N elements and their (N,N) pairwise distance matrix we would like to get the subset of S elements such that the sum of elements in the corresponding (S,S) submatrix is minimal. See example below.

```
  [0  1  2  3  4] indeces 
      i  j     k    
      │  │     │          i j k   = [1, 2, 4]
   0  1  6  4  1                   
i──1  0  3  1  7       i  0 3 7     
j──6  3  0  2  3  -->  j  3 0 3  -->  7 + 3 + 3 = 13 👎
   4  1  2  0  1       k  7 3 0
k──1  7  3  1  0

         i  j  k    
         │  │  │          i j k  = [2, 3, 4]   
   0  1  6  4  1                   
   1  0  3  1  7       i  0 2 3     
i──6  3  0  2  3  -->  j  2 0 1  -->  2 + 1 + 3 = 6 👍
j──4  1  2  0  1       k  3 1 0
k──1  7  3  1  0
```

All the possible subsets are ${N}\choose{S}$ and for N = 1024, S = 20 (like in the tests) we would have to check ${1024}\choose{20}$ $= 5.479 \times 10^{41}$ of them. 

A few too many. Instead we are going to use an evolutionary approach to search for it.

# Example usage

The usage is quite straight forward since there are only a couple of functions exported `pairwise_cosine` and `extract`.

```python
>>> import matplotlib.pyplot as plt
>>> from subpair import pairwise_cosine
>>>
>>> X = np.random.rand(N, K).astype(np.float32)
>>> distances = pairwise_cosine(X) # (N,N)
>>> ...
>>> best, stats = extract(distances, P=200, S=S, K=50, M=3, O=2, its=3_000)
100%|█████████████████████████████████| 3000/3000 [00:03<00:00, 817.42it/s]
>>> plt.plot(stats["fits"]); plt.show()
```
<p align="left">
    <img width="500" alt="Logo" src="https://user-images.githubusercontent.com/3115640/204059389-730df61a-4e87-4023-b7c7-038b329dc6a6.png">
    <p>(We have sprinkled a few negative numbers to see if the algorithm can find them)</p>
</p>
Where the options of extract are parameters for the evolutionary algorithm:

``` 
distances (int, int) : N vectors of length L
        P (int)      : population size
        S (int)      : desired subset size <- determines size of output
        K (int)      : number of parents (P-K children)
        M (int)      : number of mutations
        O (int)      : fraction of crossovers e.g. O=2 -> 1/2, O=10 -> 1/10, (bigger=faster)
```

# Note

This repo contains both numpy and numba/CUDA versions of the pairwise cosine distance matrix calculation. But numpy is already _blazingly_ fast so the cuda version is provided mostly for inspiration. Our numpy version is very similar to sklearn's [metrics.pairwise.cosine_distances](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_distances.html) but slightly faster. Sklearn's one has some extra nicities that our simplified version does not have.

```bash
> python flops.py # On Macbook pro M1 Max
N=513 K=2304 GOPs=1
  sklearn: 0.01s - 109.4 GFLOPS
    numpy: 0.00s - 162.4 GFLOPS

N=1027 K=2304 GOPs=2
  sklearn: 0.02s - 135.9 GFLOPS
    numpy: 0.01s - 192.4 GFLOPS

N=2055 K=2304 GOPs=10
  sklearn: 0.07s - 142.9 GFLOPS
    numpy: 0.06s - 166.0 GFLOPS

N=4111 K=2304 GOPs=39
  sklearn: 0.20s - 195.8 GFLOPS
    numpy: 0.16s - 248.6 GFLOPS

N=8223 K=2304 GOPs=156
  sklearn: 0.61s - 255.3 GFLOPS
    numpy: 0.54s - 289.5 GFLOPS

N=16447 K=2304 GOPs=623
  sklearn: 2.11s - 295.4 GFLOPS
    numpy: 1.79s - 347.9 GFLOPS
```

# Todo
- [ ] Add type info to minimize.py to allow for AOT compilation.
