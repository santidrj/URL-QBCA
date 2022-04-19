import os
import time

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    fowlkes_mallows_score,
)

from source.qbca import QBCA
from source.utils import dunn_index

N_SAMPLES = 10000
SEED = 2022

fig_dir = "../figures"
if not os.path.exists(fig_dir):
    os.mkdir(fig_dir)


def create_gaussian_5():
    return make_blobs(
        n_samples=N_SAMPLES, centers=5, cluster_std=0.06, random_state=SEED
    )


def create_gaussian_25():
    return make_blobs(
        n_samples=N_SAMPLES, centers=25, cluster_std=0.02, random_state=SEED
    )


def initialize_algorithms(k, threshold, max_iter):
    qbca = QBCA(n_clusters=k, threshold=threshold, max_iter=max_iter)
    kmeans = KMeans(
        n_clusters=k, init="random", max_iter=max_iter, tol=threshold, algorithm="full"
    )
    kmeans_plusplus = KMeans(
        n_clusters=k,
        init="k-means++",
        max_iter=max_iter,
        tol=threshold,
        algorithm="full",
    )
    minibatch_kmeans = MiniBatchKMeans(n_clusters=k, max_iter=max_iter)

    return {
        "qbca": qbca,
        "kmeans": kmeans,
        "kmeans_plusplus": kmeans_plusplus,
        "minibatch_kmeans": minibatch_kmeans,
    }


def save_metrics(out_file, metrics):
    performance_metrics = [
        "Training time",
        "#Distance computations"
    ]

    external_metrics = [
        "Adjusted Mutual Information",
        # "Adjusted Random Score",
        "Fowlkes-Mallows Score"
    ]

    internal_metrics = [
        "Calinski-Harabasz Index",
        # "Davies-Bouldin Index",
        "Dunn Index"
    ]

    out_dir = "../out"

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    out_file = os.path.join(out_dir, out_file)
    s = metrics[performance_metrics].style
    s.format('{:.4f}')
    s.to_latex(f"{out_file}_performance.tex", position='htbp', position_float='centering', hrules=True)

    s = metrics[external_metrics].style.highlight_max(props='textbf: --rwrap', axis=1)
    s.format('{:.4f}')
    s.to_latex(f"{out_file}_external_metrics.tex", position='htbp', position_float='centering', hrules=True)

    s = metrics[internal_metrics].style.highlight_max(props='textbf: --rwrap', axis=1)
    s.format('{:.4f}')
    s.to_latex(f"{out_file}_internal_metrics.tex", position='htbp', position_float='centering', hrules=True)


def run_test(fig_name, algorithms, data, gs, n_iter=10, verbose=False):
    metrics = {}

    df = pd.DataFrame(
        columns=[
            "Training time",
            "#Distance computations",
            "Adjusted Mutual Information",
            "Adjusted Random Score",
            "Fowlkes-Mallows Score",
            "Calinski-Harabasz Index",
            "Davies-Bouldin Index",
            "Dunn Index"
        ]
    )

    fig, ax = plt.subplots(2, 2)

    for i, (name, algorithm) in enumerate(algorithms.items()):
        print(f'Start running {name}')
        avg_time = 0
        avg_ami = 0
        avg_ar = 0
        avg_fowlkes = 0
        avg_calinski = 0
        avg_davies = 0
        avg_dunn = 0
        avg_dist_count = 0
        for _ in range(n_iter):
            start = time.perf_counter()
            algorithm.fit(data)
            avg_time += time.perf_counter() - start
            labels = algorithm.predict(data)

            avg_ami += adjusted_mutual_info_score(gs, labels)
            avg_ar += adjusted_rand_score(gs, labels)
            avg_fowlkes += fowlkes_mallows_score(gs, labels)
            avg_calinski += calinski_harabasz_score(data, labels)
            avg_davies += davies_bouldin_score(data, labels)
            avg_dunn += dunn_index(data, labels)

            if hasattr(algorithm, "n_dist_"):
                avg_dist_count += algorithm.n_dist_
            else:
                avg_dist_count += algorithm.n_iter_ * data.shape[0]

        idx = np.unravel_index(i, (2, 2))
        ax[idx].scatter(data[:, 0], data[:, 1], c=labels)
        ax[idx].set_title(f'After {name}')

        avg_time /= n_iter
        avg_ami /= n_iter
        avg_ar /= n_iter
        avg_fowlkes /= n_iter
        avg_calinski /= n_iter
        avg_davies /= n_iter
        avg_dunn /= n_iter
        avg_dist_count /= n_iter

        if verbose:
            out_string = f"""{name}
Training time: {avg_time:.4f} seconds

Number of distance computations: {avg_dist_count}

External validation:
    Adjusted Mutual Information: {avg_ami:.4f}
    Adjusted Random Score: {avg_ar:.4f}
    Fowlkes-Mallows Score: {avg_fowlkes:.4f}
Internal validation:
    Calinski-Harabasz Index: {avg_calinski:.4f}
    Davies-Bouldin Index: {avg_davies:.4f}
    Dunn Index: {avg_dunn:.4f}
    """
            print(out_string)

        aux_df = pd.DataFrame(
            [[
                avg_time,
                avg_dist_count,
                avg_ami,
                avg_ar,
                avg_fowlkes,
                avg_calinski,
                avg_davies,
                avg_dunn
            ]],
            columns=df.columns,
            index=[name]
        )
        df = pd.concat([df, aux_df])
        # metrics[name] = {
        #     "time": avg_time,
        #     "dist_count": avg_dist_count,
        #     "external": [avg_ami, avg_ar, avg_fowlkes],
        #     "internal": [avg_calinski, avg_davies],
        # }

    fig.savefig(os.path.join(fig_dir, f"{fig_name}.png"))

    return df
