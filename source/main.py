import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA

from source.qbca import QBCA
from source.utils import read_datafile

plt.style.use("ggplot")

# file = "test.csv"
# data, gs = read_datafile(file)
iris = datasets.load_iris()
data, gs = iris.data, iris.target

if __name__ == "__main__":
    x, y = QBCA(3, 0.0001, 50).fit_predict(data.copy())
    if data.shape[1] > 2:
        p = PCA(n_components=2)
        data = p.fit_transform(data)
        x = p.transform(x)
    fig, axs = plt.subplots(2, 1)
    axs[0].set_title("Original")
    axs[0].scatter(data[:, 0], data[:, 1], c=gs)
    axs[1].set_title("After QBCA")
    axs[1].scatter(x[:, 0], x[:, 1], c=y)
    plt.tight_layout()
    plt.show()
