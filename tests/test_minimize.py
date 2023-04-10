from subpair import *
import numpy as np


def test_min_subset():

    N = 1024
    K = 512

    X = np.random.rand(N, K).astype(np.float32)
    distances = pairwise_cosine(X)

    # import matplotlib.pyplot as plt
    # plt.imshow(distances, interpolation="nearest")
    # plt.show()

    S = 20
    locs = np.arange(N)
    np.random.shuffle(locs)
    xs = locs[:S]
    for i in xs:
        for j in xs:
            distances[i, j] = -1

    print()
    best, stats = extract(distances, P=200, S=S, K=50, M=3, R=0.95, O=2, its=3_000)
    # print("Final fit:", stats["fits"][-1])

    # import matplotlib.pyplot as plt
    # plt.plot(stats["fits"])
    # plt.grid()
    # plt.ylabel("Subset distances")
    # plt.xlabel("Iterations")
    # plt.title("Minimal subset extraction (lower is better)")
    # plt.tight_layout()
    # plt.savefig("min_subset.png", dpi=300)
    # plt.show()

    missed = len(set(best).difference(set(xs)))
    accuracy = 1 - missed / len(best)

    print("Accuracy:", accuracy)

    assert accuracy > 0.5
