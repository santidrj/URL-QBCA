import heapq
import math

import numpy as np


class QBCA:
    def __init__(self, n_seeds):
        self.n_seeds = n_seeds

    def fit(self, x):
        self._quantization(x)
        self._cci(self.n_seeds)
        pass

    def predict(self, x):
        pass

    def fit_predict(self, x):
        pass

    def _initialize_bins(self, x):
        self.m_dims = x.shape[-1]
        self.n_bins = math.floor(math.log(x.shape[0], self.m_dims))
        self.bin_size = (np.max(x, axis=0) - np.min(x, axis=0)) / self.n_bins
        self.bins = np.zeros(self.n_bins ** self.m_dims, dtype=int)
        self.min_values = np.min(x, axis=0)

    def _quantize_point(self, point):
        epsilon = point.copy()
        mask = point == self.min_values
        epsilon[mask] = 1
        epsilon[~mask] = point[~mask] - self.min_values[~mask]
        epsilon = np.ceil(epsilon / self.bin_size)
        p = np.full_like(epsilon, self.bin_size)
        exp = np.arange(self.m_dims - 1, -1, -1)
        return np.sum((epsilon - 1) * (p ** exp) + epsilon[-1]).astype(int)

    def _quantization(self, x):
        self._initialize_bins(x)
        idxs = np.apply_along_axis(self._quantize_point, axis=1, arr=x)
        for idx in idxs:
            self.bins[idxs] += 1

    def _get_non_zero_bins(self):
        non_zero_bins = np.where(self.bins != 0)
        return self.bins[non_zero_bins], non_zero_bins[0]

    def _cci(self, n_seeds):
        seed_list = np.zeros((n_seeds, self.m_dims))
        non_zero_bins, non_zero_bins_idx = self._get_non_zero_bins()

        # Negate non_zero_bins values to simulate a max_heap
        max_heap = np.stack((-non_zero_bins, non_zero_bins_idx)).T
        heapq.heapify(max_heap.tolist())
