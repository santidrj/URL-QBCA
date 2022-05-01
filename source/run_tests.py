import os
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_wine, load_iris

from source.test_utils import (
    create_gaussian_5,
    create_gaussian_25,
    run_test,
    save_metrics,
    initialize_algorithms,
    load_image,
    run_segmentation,
)

plt.style.use("ggplot")


def test_gaussian(algorithms, out_file, option=0, verbose=False):
    if option == 0:
        print("Test Gaussian 5")
        data, gs = create_gaussian_5()
    elif option == 1:
        print("Test Gaussian 25")
        data, gs = create_gaussian_25()
    else:
        raise ValueError(f"There is no option {option}. Available options are [0, 1].")

    metrics = run_test(out_file, algorithms, data, gs, verbose=verbose)
    save_metrics(out_file, metrics)
    return metrics["#Distance computations"]


def test_wine(algorithms, out_file, verbose=False):
    print("Test Wine")
    wine = load_wine()
    metrics = run_test(out_file, algorithms, wine.data, wine.target, verbose=verbose)
    save_metrics(out_file, metrics)
    return metrics["#Distance computations"]


def test_iris(algorithms, out_file, verbose=False):
    print("Test Iris")
    iris = load_iris()
    metrics = run_test(out_file, algorithms, iris.data, iris.target, verbose=verbose)
    save_metrics(out_file, metrics)
    return metrics["#Distance computations"]


def test_image(algorithms, out_file, verbose=False):
    print("Test Butterfly image")
    data, im = load_image("butterfly")
    metrics = run_segmentation(out_file, algorithms, data, im, verbose=verbose)
    save_metrics(out_file, metrics)
    return metrics["#Distance computations"]


plot_out_1 = os.path.join("figures", "gaussian-performance.png")
plot_out_2 = os.path.join("figures", "real-data-performance.png")
plot_out_3 = os.path.join("figures", "image-performance.png")
fig, ax = plt.subplots()

dist_com1 = test_gaussian(
    initialize_algorithms(5, 1e-4, 30), "gaussian_5", option=0, verbose=True
)

dist_com2 = test_gaussian(
    initialize_algorithms(25, 1e-4, 30), "gaussian_25", option=1, verbose=True
)

dist_com3 = test_wine(initialize_algorithms(3, 1e-4, 30), "wine", verbose=True)

dist_com4 = test_iris(initialize_algorithms(3, 1e-4, 30), "iris", verbose=True)

dist_com5 = test_image(initialize_algorithms(5, 1e-2, 50), "butterfly")

df = pd.concat([dist_com1, dist_com2], axis=1)
df.columns = ["Gaussian(5)", "Gaussian(25)"]
ax = df.T.plot.bar(rot=0, title="Performance comparison", ylabel="#Distance computations")

plt.tight_layout()
plt.savefig(plot_out_1)
plt.clf()

df = pd.concat([dist_com3, dist_com4], axis=1)
df.columns = ["wine", "irirs"]
ax = df.T.plot.bar(rot=0, title="Performance comparison", ylabel="#Distance computations")

plt.tight_layout()
plt.savefig(plot_out_2)
plt.clf()

ax = dist_com5.plot.bar(rot=0, title="Performance comparison", ylabel="#Distance computations")

plt.tight_layout()
plt.savefig(plot_out_3)
plt.clf()
