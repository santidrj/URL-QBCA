import copy
import math
from itertools import product

import numpy as np
from scipy.spatial.distance import euclidean

from source.data_structures import MaxHeap


class QBCA:
    def __init__(self, n_seeds):
        self.n_seeds = n_seeds

    def fit(self, x):
        self._quantization(x)
        self._cluster_center_initialization()
        self._cluster_center_assignment()
        pass

    def predict(self, x):
        pass

    def fit_predict(self, x):
        pass

    def _initialize_bins(self, x):
        self.m_dims = x.shape[-1]
        self.min_values = np.min(x, axis=0)
        self.n_bins = math.floor(math.log(x.shape[0], self.m_dims))
        self.bin_size = (np.max(x, axis=0) - np.min(x, axis=0)) / self.n_bins
        self.bin_counts = np.zeros(self.n_bins ** self.m_dims, dtype=int)
        self.bins = [[] for _ in range(self.n_bins ** self.m_dims)]
        self.hist_shape = tuple([self.n_bins] * self.m_dims)

        # Create a mask for all the possible positions of the neighbors wrt a point
        aux_list = list(product([-1, 0, 1], repeat=self.m_dims))
        aux_list.remove(tuple(np.zeros(self.m_dims)))
        self.neighbors_mask = np.array(aux_list)

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
        for idx, point in zip(idxs, x):
            self.bin_counts[idx] += 1
            self.bins[idx].append(point)

    def _cluster_center_initialization(self):
        # Create Max-Heap with the non-zero bins
        non_empty_bins = self._get_non_empty_bins()
        max_heap = MaxHeap()
        for b in np.stack(non_empty_bins).T:
            max_heap.heappush(tuple(b))
        aux_heap = copy.copy(max_heap)
        seed_list = []

        while len(max_heap) > 0:
            b = max_heap.heappop()
            flag = True
            neighbors = self._get_neighbors(b[1])

            if (self.bin_counts[neighbors] <= b[0]).all():
                seed_list.append(b)

        # If there are not enough seeds
        if len(seed_list) < self.n_seeds:
            seed_list.extend(
                [x for x in aux_heap if x not in seed_list][
                : self.n_seeds - len(seed_list)
                ]
            )

        seed_list.sort(key=self._bin_cardinality)

        # self.seeds = np.zeros((self.n_seeds, self.m_dims))
        self.seeds = np.array(
            [
                self._compute_center(self.bins[b_idx])
                for (_, b_idx) in seed_list[: self.n_seeds]
            ]
        )
        # for s_idx, (_, b_idx) in enumerate(seed_list[:self.n_seeds]):
        #     self.seeds[s_idx] = self._compute_center(self.bins[b_idx])

    def _cluster_center_assignment(self):
        non_empty_bins, non_empty_bins_idx = self._get_non_empty_bins()
        coordinates = np.array(
            np.unravel_index(non_empty_bins_idx, self.hist_shape)
        ).transpose()
        min_coords = self.bin_size * coordinates
        max_coords = self.bin_size * (coordinates + 1)
        min_max_distances = self._compute_min_max_distances(
            max_coords, min_coords, non_empty_bins
        )
        # TODO: finish this

    def _compute_min_max_distances(self, max_coords, min_coords, non_empty_bins):
        max_distances = self._compute_max_distances(
            max_coords, min_coords, non_empty_bins
        )
        return np.min(max_distances, axis=1)

    def _compute_max_distances(self, max_coords, min_coords, non_empty_bins):
        threshold = (min_coords + max_coords) / 2
        max_distances = np.zeros((len(non_empty_bins), len(self.seeds)))
        for i, t in enumerate(threshold):
            upsilon = np.zeros(self.seeds.shape)
            mask = self.seeds >= t
            min_coord = np.tile(min_coords[i], (upsilon.shape[0], 1))
            max_coord = np.tile(max_coords[i], (upsilon.shape[0], 1))
            upsilon[mask] = min_coord[mask]
            upsilon[~mask] = max_coord[~mask]
            max_distances[i] = np.array(
                [euclidean(s, u) for s, u in zip(self.seeds, upsilon)]
            )
        return max_distances

    def _get_non_empty_bins(self):
        """
        Get the non-empty histogram bins.
        :return: A tuple with the bin cardinality and its index.
        """
        non_zero_bins = np.where(self.bin_counts != 0)
        return self.bin_counts[non_zero_bins], non_zero_bins[0]

    def _bin_cardinality(self, bin: tuple):
        return bin[0]

    def _get_neighbors(self, index):
        """
        Given a bin index return all its neighbors indexes.

        :param index: The bin index.
        :return: An ndarray of the neighbors indexes.
        """
        coord = np.unravel_index(index, self.hist_shape)
        neighbors = self.neighbors_mask + coord
        # Remove coordinates that are out of the histogram
        neighbors = neighbors[~(neighbors < 0).any(1)]
        neighbors = neighbors[~(neighbors >= self.n_bins).any(1)]
        neighbors_idx = []
        for n in neighbors:
            neighbors_idx.append(np.ravel_multi_index(n, self.hist_shape))
        return np.array(neighbors_idx)

    def _compute_center(self, cluster):
        return np.mean(np.array(cluster), axis=0)

    def _compute_maximum_distance(self, bins, seeds):
        pass
