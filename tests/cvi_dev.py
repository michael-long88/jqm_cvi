__author__ = "Joaquim Viegas"

# ==============================================================================
# Description
# ==============================================================================

import jqmcvi.basec as jqmcvi
import jqmcvi.base as jqmcvin

import numpy as np
import pickle
from timeit import timeit
from sklearn.metrics import silhouette_score as Sil

if __name__ == "__main__":
    ccs = pickle.load(open("ccs.pkl", "rb"))
    points = pickle.load(open("ps.pkl", "rb"))
    labels = pickle.load(open("lbls.pkl", "rb"))

    cluster_points = [[] for _ in range(len(ccs))]

    for label_index, label in enumerate(labels):
        cluster_points[label].append(points[label_index])
    cluster_points[0] = np.array(cluster_points[0])
    cluster_points[1] = np.array(cluster_points[1])

    print(
        timeit(
            "Sil(points, labels, metric='euclidean')",
            setup="from __main__ import Sil, points, labels",
            number=1,
        )
    )
    print(
        timeit(
            "jqmcvi.dunn(cluster_points)",
            setup="from __main__ import jqmcvi, cluster_points",
            number=1,
        )
    )
    print(
        timeit(
            "jqmcvin.dunn_fast(points, labels)",
            setup="from __main__ import jqmcvin, points, labels",
            number=1,
        )
    )
    print(
        timeit(
            "jqmcvi.davisbouldin(cluster_points, ccs)",
            setup="from __main__ import jqmcvi, cluster_points, ccs",
            number=1,
        )
    )
    print(
        timeit(
            "jqmcvin.davisbouldin(cluster_points, ccs)",
            setup="from __main__ import jqmcvin, cluster_points, ccs",
            number=1,
        )
    )

    print(Sil(points, labels, metric="euclidean"))
    print(jqmcvi.dunn(cluster_points))
    print(jqmcvin.dunn_fast(points, labels))
    print(jqmcvi.davisbouldin(cluster_points, ccs))
    print(jqmcvin.davisbouldin(cluster_points, ccs))
