import time
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score, fowlkes_mallows_score, normalized_mutual_info_score

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


def test_gaussian(option = 0):
    if option == 0:
        print("Test Gaussian 5")
        data, gs = create_gaussian_5()
        algorithms = initialize_algorithms(5, 1e-4, 300)
    elif option == 1:
        print("Test Gaussian 25")
        data, gs = create_gaussian_25()
        algorithms = initialize_algorithms(25, 1e-4, 300)
    else:
        raise ValueError(f"There is no option {option}. Available options are [0, 1].")

    metrics = {}

    for name, algorithm in algorithms.items():
        start = time.perf_counter()
        algorithm.fit(data)
        end = round(time.perf_counter() - start, 4)
        labels = algorithm.predict(data)
        
        ami = adjusted_mutual_info_score(gs, labels)
        ar = adjusted_rand_score(gs, labels)
        fowlkes = fowlkes_mallows_score(gs, labels)
        calinski = calinski_harabasz_score(data, labels)
        davies = davies_bouldin_score(data, labels)
        out_string = f"""{name}
Training time: {end} seconds

External validation:
    Adjusted Mutual Information: {ami:.4f}
    Adjusted Random Score: {ar:.4f}
    Fowlkes-Mallows Score: {fowlkes:.4f}

Internal validation:
    Calinski-Harabasz Index: {calinski:.4f}
    Davies-Bouldin Index: {davies:.4f}
"""
        print(out_string)
        metrics[name] = {'time': end, 'external': [ami,ar,fowlkes], 'internal': [calinski, davies]}
    
    return metrics

test_gaussian(0)
test_gaussian(1)