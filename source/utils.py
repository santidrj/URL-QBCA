import os.path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA


def read_datafile(filepath, n_dims=-1, has_class=True):
    df = pd.read_csv(filepath)
    if has_class:
        x, y = df.iloc[:, :-1].to_numpy(copy=True), df.iloc[:, -1].to_numpy(copy=True)
    else:
        x, y = df.to_numpy(copy=True), None

    if 0 < n_dims < x.shape[1]:
        pca = PCA(n_components=n_dims)
        x = pca.fit_transform(x)

    return x, y


def clusters_dissimilarity(c1, c2):
    """Calculate dissimilarity metric between two clusters"""
    return np.min(cdist(c1, c2))


def cluster_diameter(cluster):
    """Compute the diameter of a cluster"""
    return np.max(cdist(cluster, cluster))


def dunn_index(X, labels):
    """Compue dunn index from cluster datapoints"""
    clusters = [X[labels == i, :] for i in np.unique(labels)]

    min_ratio = None
    max_diameter = np.max([cluster_diameter(c) for c in clusters])
    k = len(clusters)
    for i in range(0, k):
        for j in range(i + 1, k):
            dissimilarity = clusters_dissimilarity(clusters[i], clusters[j])
            ratio = dissimilarity / max_diameter
            if not min_ratio or ratio < min_ratio:
                min_ratio = ratio
    return min_ratio
