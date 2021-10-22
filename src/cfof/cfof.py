from multiprocessing import Pool
from typing import Callable, List, Union

import numpy as np
from sklearn.neighbors import NearestNeighbors


class CFOF:
    """
    Concentration Free Outlier Factor.

    Parameters
    ----------
    metric : str or callable, default 'euclidean'
        Must be a valid `sklearn.metrics`.
    rhos : List[float], default [0.01]
        `ϱ` parameters, fraction of the data population.
        Must be between 0 and 1.
    n_jobs : int, default None
        The number of jobs to use for the computation.
        `-1` means using all processors.

    References
    ----------
    .. [1] Angiulli, F. (2020, January).
           CFOF: A Concentration Free Measure for Anomaly Detection.
           In ACM Transactions on Knowledge Discovery from Data.
    """
    def __init__(self,
                 metric: Union[str, Callable] = 'euclidean',
                 rhos: List[float] = [0.01],
                 n_jobs=None) -> None:
        self.rhos = np.array(rhos)
        if (self.rhos < 0.0).any() or (self.rhos > 1.0).any():
            raise ValueError(f'rhos ({self.rhos}) must be between 0 and 1')

        self.metric = metric
        self.n_jobs = n_jobs

        self.thresholds = None
        self.n = None

    def compute(self, X: np.ndarray) -> np.ndarray:
        """
        Compute hard-CFOF scores.

        Parameters
        ----------
        X : numpy.ndarray
            Dataset.

        Returns
        -------
        numpy.ndarray
            CFOF scores `sc`.
            sc[i, l] is score of object `i` for `ϱl` (rhos[l]).
        """
        self._update_params(X)
        neighbors = self._find_neighbors(X)
        return self._cfof(neighbors)

    def compute_from_distance_matrix(
            self, distance_matrix: np.ndarray) -> np.ndarray:
        """
        Compute hard-CFOF scores from distance matrix.

        Parameters
        ----------
        compute_from_distance_matrix : numpy.ndarray
            Distance matrix.

        Returns
        -------
        numpy.ndarray
            CFOF scores `sc`.
            sc[i, l] is score of object `i` for `ϱl` (rhos[l]).
        """
        self._update_params(distance_matrix)
        neighbors = self._find_neighbors_from_distance_matrix(distance_matrix)
        return self._cfof(neighbors)

    def _update_params(self, input: np.ndarray):
        self.n, _ = input.shape
        self.thresholds = self.rhos * self.n

    # `neighbors[i]` are the neighbours of i in order of proximity.
    # Every point is the first neighbor of itself.
    def _cfof(self, neighbors: np.ndarray) -> np.ndarray:
        # `min_k_neighborhood[i, j]` represents min k such that i contains j
        # in its neighborhood.
        min_k_neighborhood = np.argsort(neighbors) + 1

        with Pool(processes=self.n_jobs) as pool:
            sc = pool.map(self._compute_col_cfof, min_k_neighborhood.T)

        sc = np.array(sc) / self.n

        return sc

    def _find_neighbors(self, X: np.ndarray, algorithm='auto'):
        nbrs = NearestNeighbors(n_neighbors=len(X),
                                algorithm=algorithm,
                                metric=self.metric,
                                n_jobs=self.n_jobs).fit(X)
        indices = nbrs.kneighbors(X, return_distance=False)
        return indices

    def _find_neighbors_from_distance_matrix(self,
                                             distance_matrix: np.ndarray):
        indices = np.argsort(distance_matrix)
        return indices

    def _compute_col_cfof(self, col):
        counter = np.bincount(col).cumsum()
        return np.array([np.argmax(counter >= t) for t in self.thresholds])
