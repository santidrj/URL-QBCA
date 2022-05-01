# URL - Quantization-Based Clustering Algorithm
To run the paper experiments execute the `run_tests.py` file in this directory.

The algorithm has been implemented to follow the `sklearn` API and can be used as follows:
```python
from sklearn import datasets
from source.qbca import QBCA

iris = datasets.load_iris()
X = iris.data
y = iris.target
q = QBCA(n_clusters=3)
predictions = q.fit_predict(X)

```

The project dependencies are listed in the `pyproject.toml` file and can be installed using the [Poetry](https://python-poetry.org/ "Poetry - Python dependency management and packaging made easy") dependency manager or any other dependency manager.