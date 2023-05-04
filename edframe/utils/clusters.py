from __future__ import annotations

from kneed import KneeLocator
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

import numpy as np


def centroids(
    X: np.ndarray,
    y: np.ndarray,
    median: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    if len(X) != len(y):
        raise ValueError
    centroids = []
    y_new = []
    knn = NearestNeighbors()
    for k in np.unique(y):
        X_class = X[y == k]
        knn.fit(X_class)
        d, _ = knn.kneighbors(X_class)
        d = np.sort(d, axis=0)[:, 1]
        kloc = KneeLocator(np.arange(len(d)), d, curve='convex')
        dbscan = DBSCAN(eps=d[kloc.knee])
        labels = dbscan.fit_predict(X_class)
        clusters = np.unique(labels)
        if len(clusters) > 1:
            X_class = X_class[labels != -1]
            labels = labels[labels != -1]
            clusters = clusters[clusters != -1]
        for cluster in clusters:
            X_cluster = X_class[labels == cluster]
            if median:
                centroid = np.median(X_cluster, axis=0)
            else:
                centroid = np.mean(X_cluster, axis=0)
            centroids.append(centroid)
        y_new.append([k] * len(clusters))
    centroids = np.stack(centroids, axis=0)
    y_new = np.asarray(y_new)
    return centroids, y_new
