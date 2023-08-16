from __future__ import annotations

from kneed import KneeLocator
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

import numpy as np


def centroids(
    X: np.ndarray,
    y: np.ndarray,
    median: bool = True,
    return_labels:bool=False,
) -> tuple[np.ndarray, np.ndarray]:
    if len(X) != len(y):
        raise ValueError

    X_centroids = []
    y_centroids= []
    knn = NearestNeighbors()

    # for k in np.unique(y):
        # X_class = X[y == k]

    knn.fit(X)
    d, _ = knn.kneighbors(X)
    d = np.sort(d, axis=0)[:, 1]
    kloc = KneeLocator(np.arange(len(d)), d, curve='convex')
    dbscan = DBSCAN(eps=d[kloc.knee])
    y_new = dbscan.fit_predict(X)
    clusters = np.unique(y_new)

    # if len(clusters) > 1:
        # X = X[y_new != -1]
        # y_new = y_new[y_new != -1]
        # clusters = clusters[clusters != -1]

    # _y = []
    # _X = []

    for cluster in clusters:
        X_cluster = X[y_new == cluster]

        if median:
            centroid = np.median(X_cluster, axis=0)
        else:
            centroid = np.mean(X_cluster, axis=0)

        X_centroids.append(centroid)
        y_centroids.append(cluster)

    X_centroids = np.asarray(X_centroids)
    y_centroids = np.asarray(y_centroids)

    if return_labels:
        return X_centroids, y_centroids, y_new

    return X_centroids, y_centroids
