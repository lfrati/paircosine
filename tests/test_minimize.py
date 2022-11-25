from subpair import *


def test_min_subset():

    # N = 1000  # population size
    # M = 20  # dna size
    # k = 200  # how many get to fuck (out of N)

    N = 1024
    distances = np.random.rand(N, N).astype(np.float32)
    distances = np.maximum(distances, distances.transpose())

    # import matplotlib.pyplot as plt
    # plt.imshow(distances, interpolation="nearest")
    # plt.show()

    S = 16
    locs = np.arange(N)
    np.random.shuffle(locs)
    xs = locs[:S]
    for i in xs:
        for j in xs:
            distances[i, j] = -1

    best, _ = select_best(distances, P=200, S=S, K=50, M=3, O=2, its=3_000)
    # print("Final fit:", stats["fits"][-1])

    # import matplotlib.pyplot as plt
    # plt.plot(stats["fits"])
    # plt.show()

    missed = len(set(best).difference(set(xs)))
    accuracy = 1 - missed / len(best)
    print(accuracy)

    assert accuracy > 0.5
