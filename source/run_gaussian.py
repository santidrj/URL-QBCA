from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, normalized_mutual_info_score

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
    elif option == 1:
        print("Test Gaussian 25")
        data, gs = create_gaussian_25()
    else:
        raise ValueError(f"There is no option {option}. Available options are [0, 1].")

    algorithms = initialize_algorithms(5, 1e-4, 300)
    scores = {}

    for name, algorithm in algorithms.items():
        if hasattr(algorithm, 'fit_predict'):
            labels = algorithm.fit_predict(data)
        else:
            algorithm.fit(data)
            labels = algorithm.predict(data)
        
        ami = adjusted_mutual_info_score(gs, labels)
        ar = adjusted_rand_score(gs, labels)
        nmi = normalized_mutual_info_score(gs, labels)
        out_string = f"""{name}
Adjusted Mutual Information: {ami}
Adjusted Random Score: {ar}
Normalized Mutual Information: {nmi}
"""
        print(out_string)
        scores[name] = [ami,ar,nmi]
    
    return scores

test_gaussian(0)
test_gaussian(1)