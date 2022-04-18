import time
import pandas as pd
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


N_SAMPLES = 10000
SEED = 2022


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


def test_gaussian(algorithms, option=0):
    if option == 0:
        print("Test Gaussian 5")
        data, gs = create_gaussian_5()
    elif option == 1:
        print("Test Gaussian 25")
        data, gs = create_gaussian_25()
    else:
        raise ValueError(f"There is no option {option}. Available options are [0, 1].")

    metrics = run_test(algorithms, data, gs)

    metrics.style.to_latex()
    return metrics


def run_test(algorithms, data, gs, n_iter=10):
    metrics = {}

    df = pd.DataFrame(
        columns=[
            "Training time",
            "#Distance computations",
            "Adjusted Mutual Information",
            "Adjusted Random Score",
            "Fowlkes-Mallows Score",
            "Calinski-Harabasz Index",
            "Davies-Bouldin Index"
        ]
    )
    for name, algorithm in algorithms.items():
        avg_time = 0
        avg_ami = 0
        avg_ar = 0
        avg_fowlkes = 0
        avg_calinski = 0
        avg_davies = 0
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

            if hasattr(algorithm, "n_dist_"):
                avg_dist_count += algorithm.n_dist_
            else:
                avg_dist_count += algorithm.n_iter_ * data.shape[0]

        avg_time /= n_iter
        avg_ami /= n_iter
        avg_ar /= n_iter
        avg_fowlkes /= n_iter
        avg_calinski /= n_iter
        avg_davies /= n_iter
        avg_dist_count /= n_iter

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
                ]],
                columns=df.columns,
                index=[name]
            )
        df = pd.concat([df,aux_df])
        metrics[name] = {
            "time": avg_time,
            "dist_count": avg_dist_count,
            "external": [avg_ami, avg_ar, avg_fowlkes],
            "internal": [avg_calinski, avg_davies],
        }
    return df


test_gaussian(initialize_algorithms(5, 1e-4, 300), 0)

test_gaussian(initialize_algorithms(25, 1e-4, 300), 1)
