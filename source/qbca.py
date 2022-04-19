import copy
import math
from itertools import product

import numpy as np
from scipy.spatial.distance import cdist, euclidean

from source.data_structures import MaxHeap


class QBCA:
    def __init__(self, n_clusters=2, threshold=0.0001, max_iter=300):
        self.n_seeds = n_clusters
        self.threshold = threshold
        self.max_iter = max_iter
        self.n_dist_ = 0

    def fit(self, X):
        x = X.copy()
        self._quantization(x)
        self._cluster_center_initialization(x)
        delta = self.threshold + 1
        n_iter = 0
        while delta > self.threshold and n_iter < self.max_iter:
            self._cluster_center_assignment(x)
            self._recompute_cluster_centers()
            delta = self._compute_termination_criteria()
            n_iter += 1

        return self

    def predict(self, X):
        x = X.copy()
        y = np.full(X.shape[0], fill_value=-1, dtype=int)
        for seed, points in enumerate(self.seed_point_indices):
            if points:
                y[points] = seed
        return y

    def fit_predict(self, X):
        self.fit(X)
        return self.predict(X)

    def _initialize_bins(self, X):
        self.m_dims = X.shape[-1]
        self.min_values = np.min(X, axis=0)
        self.bins_per_dim = math.floor(math.log(X.shape[0], self.m_dims))
        self.bin_size = (np.max(X, axis=0) - np.min(X, axis=0)) / self.bins_per_dim
        self.bins_cardinality = np.zeros(self.bins_per_dim**self.m_dims, dtype=int)
        self.bins = [[] for _ in range(self.bins_per_dim**self.m_dims)]
        self.hist_shape = tuple([self.bins_per_dim] * self.m_dims)

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
        exp = np.arange(self.m_dims - 1, -1, -1)
        bin_idx = np.sum((epsilon - 1) * (self.bins_per_dim**exp)).astype(int)
        return bin_idx

    def _quantization(self, X):
        self._initialize_bins(X)
        idxs = np.apply_along_axis(self._quantize_point, axis=1, arr=X)
        for point, idx in enumerate(idxs):
            self.bins_cardinality[idx] += 1
            self.bins[idx].append(point)

    def _cluster_center_initialization(self, X):
        # Create Max-Heap with the non-zero bins
        non_empty_bins = self._get_non_empty_bins()
        max_heap = MaxHeap()
        for b in np.stack(non_empty_bins).T:
            max_heap.heappush(tuple(b))
        aux_heap = copy.deepcopy(max_heap)
        seed_list = []

        while len(max_heap) > 0:
            b = max_heap.heappop()
            flag = True
            neighbors = self._get_neighbors(b[1])

            if (self.bins_cardinality[neighbors] <= b[0]).all():
                seed_list.append(b)

        # If there are not enough seeds
        if len(seed_list) < self.n_seeds:
            seed_list.extend(
                [x for x in aux_heap if x not in seed_list][
                    : self.n_seeds - len(seed_list)
                ]
            )

        seed_list.sort(key=self._bin_cardinality)

        self.seeds = np.array(
            [
                self._compute_center(X[self.bins[b_idx]])
                for (_, b_idx) in seed_list[: self.n_seeds]
                if self.bins[b_idx]
            ]
        )

    def _cluster_center_assignment(self, X):
        _, non_empty_bins_idx = self._get_non_empty_bins()
        coordinates = np.array(
            np.unravel_index(non_empty_bins_idx, self.hist_shape)
        ).transpose()
        min_coords = self.min_values + self.bin_size * coordinates
        max_coords = self.min_values + self.bin_size * (coordinates + 1)
        min_max_distances = self._compute_min_max_distances(max_coords, min_coords)
        min_distances = self._compute_min_distances(max_coords, min_coords)

        candidates = (min_distances.transpose() <= min_max_distances).transpose()
        self.seed_points = [[] for _ in range(len(self.seeds))]
        self.seed_point_indices = [[] for _ in range(len(self.seeds))]
        for i, idx in enumerate(non_empty_bins_idx):
            points = np.array(X[self.bins[idx]])
            distances = cdist(points, self.seeds[candidates[i]])
            candidates_idx = np.where(candidates[i])[0]
            closest_seed = np.argmin(distances, axis=1)
            for j, cs in enumerate(closest_seed):
                self.seed_points[candidates_idx[cs]].append(points[j])
                self.seed_point_indices[candidates_idx[cs]].append(self.bins[idx][j])

    def _compute_min_max_distances(self, max_coords, min_coords):
        max_distances = self._compute_max_distances(max_coords, min_coords)
        return np.min(max_distances, axis=1)

    def _compute_max_distances(self, max_coords, min_coords):
        threshold = (min_coords + max_coords) / 2
        max_distances = np.zeros((len(max_coords), len(self.seeds)))
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
        self.n_dist_ += max_distances.size
        return max_distances
    

    def _compute_min_distances(self, max_coords, min_coords):
        min_distances = np.zeros((len(max_coords), len(self.seeds)))
        for i, (mx, mn) in enumerate(zip(max_coords, min_coords)):
            lb = self.seeds.copy()
            mask_1 = self.seeds < mn
            mask_2 = self.seeds > mx
            min_coord = np.tile(mn, (lb.shape[0], 1))
            max_coord = np.tile(mx, (lb.shape[0], 1))
            lb[mask_1] = min_coord[mask_1]
            lb[mask_2] = max_coord[mask_2]
            min_distances[i] = np.array(
                [euclidean(s, u) for s, u in zip(self.seeds, lb)]
            )

        self.n_dist_ += min_distances.size
        return min_distances


    def _get_non_empty_bins(self):
        """
        Get the non-empty histogram bins.

        :return: A tuple with the bin cardinality and its index.
        """
        non_zero_bins = np.where(self.bins_cardinality != 0)
        return self.bins_cardinality[non_zero_bins], non_zero_bins[0]

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
        neighbors = neighbors[~(neighbors >= self.bins_per_dim).any(1)]
        neighbors_idx = []
        for n in neighbors:
            neighbors_idx.append(np.ravel_multi_index(n, self.hist_shape))
        return np.array(neighbors_idx)

    def _compute_center(self, cluster):
        return np.mean(np.array(cluster), axis=0)

    def _recompute_cluster_centers(self):
        self.old_seeds = self.seeds.copy()
        for idx, _ in enumerate(self.seed_points):
            if self.seed_points[idx]:
                self.seeds[idx] = self._compute_center(self.seed_points[idx])

    def _compute_termination_criteria(self):
        phi = ((self.old_seeds - self.seeds) ** 2).sum(axis=1)
        return phi.sum() / self.n_seeds
