from typing import Callable, List, Optional, Union

import numpy as np
from sklearn.metrics import pairwise_distances


class FastCFOF:
    """
    Fast-CFOF.

    Fast-CFOF exploits sampling to avoid the computation of exact nearest
    neighbors. The cost of fast-CFOF is linear both in the dataset size and
    dimensionality.

    Parameters
    ----------
    metric : str or callable, default 'euclidean'
        Must be a valid `sklearn.metrics`.
    rhos : List[float], default [0.001, 0.005, 0.01, 0.05, 0.1]
        `ϱ` parameters, fraction of the data population.
        Must be between 0 and 1.
    epsilon : float, default 0.01
        ϵ, absolute error. (0 < ϵ < 1)
    delta : float, default 0.01
        δ, error probability. (0 < δ < 1)
    c : int, default 1
        c, with c ∈ [0, 3].
    n_bins : int, default 10
        Histogram bins.
    partition_size : Optional[int], default None
        Size of partitions.
        If None, it is computed using ϵ and δ.
        If given, ϵ and δ are ignored.
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
                 rhos: List[float] = [0.001, 0.005, 0.01, 0.05, 0.1],
                 epsilon: float = 0.01,
                 delta: float = 0.01,
                 c: int = 1,
                 n_bins: int = 10,
                 partition_size: Optional[int] = None,
                 n_jobs=None) -> None:
        self.metric = metric

        self.rhos = np.array(rhos)
        if (self.rhos < 0.0).any() or (self.rhos > 1.0).any():
            raise ValueError(f'rhos ({self.rhos}) must be between 0 and 1')

        if not (0 < epsilon < 1):
            raise ValueError(f'epsilon ({epsilon}) must be between 0 and 1')

        if not (0 < delta < 1):
            raise ValueError(f'delta ({delta}) must be between 0 and 1')

        if not (0 <= c <= 3):
            raise ValueError(f'c ({c}) must be between 0 and 3')
        self.c = c

        if not n_bins > 0:
            raise ValueError(f'n_bins ({n_bins}) must be positive')
        self.n_bins = n_bins

        if partition_size is None:
            self.s = int(np.ceil((1 / (2 * (epsilon**2))) * np.log(2 / delta)))
        else:
            self.s = partition_size

        self.n_jobs = n_jobs

        self.binning_ratio = None
        self.binning_ratio_log = None
        self.n = None

        self.X = None
        self.distance_matrix = None

        # sc[i, l] is score of object `i` for `ϱl` (rhos[l]).
        self.sc = None

    def compute(self, X: np.ndarray) -> np.ndarray:
        """
        Compute soft-CFOF scores.

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
        self.X = X
        self._fast_cfof()
        self.X = None
        return self.sc

    def compute_from_distance_matrix(
            self, distance_matrix: np.ndarray) -> np.ndarray:
        """
        Compute hard-CFOF scores from distance matrix.

        Parameters
        ----------
        distance_matrix : numpy.ndarray
            Distance matrix.

        Returns
        -------
        numpy.ndarray
            CFOF scores `sc`.
            sc[i, l] is score of object `i` for `ϱl` (rhos[l]).
        """
        self.distance_matrix = distance_matrix
        self._fast_cfof()
        self.distance_matrix = None
        return self.sc

    def _update_params(self):
        if self.X is not None:
            self.n, _ = self.X.shape
        elif self.distance_matrix is not None:
            self.n, _ = self.distance_matrix.shape

        self.binning_ratio = self.n**(1 / (self.n_bins - 1))
        self.binning_ratio_log = np.log(self.binning_ratio)
        self.sc = np.zeros((self.n, len(self.rhos)))

    def _fast_cfof(self):
        self._update_params()

        i = 0

        if self.s > self.n:
            raise ValueError(
                f"Partition (s = {self.s}) can't be bigger than dataset (n = {self.n})"
            )

        while i < self.n:
            if i + self.s < self.n:
                a = i
            else:
                a = self.n - self.s
            b = a + self.s
            self._fast_cfof_part(start_i=a, end_i=b)
            i = i + self.s

    def _fast_cfof_part(self, start_i: int, end_i: int):
        if self.X is not None:
            # Distances computation
            distances = pairwise_distances(self.X[start_i:end_i],
                                           n_jobs=self.n_jobs)
        elif self.distance_matrix is not None:
            distances = self.distance_matrix[start_i:end_i, start_i:end_i]

        s = end_i - start_i

        hst = np.zeros((s, self.n_bins))

        # Nearest neighbor count estimation
        for i in range(s):
            dst = distances[i, :]

            # Count update
            ord = np.argsort(dst)

            p_s = (np.arange(s) + 1) / s
            k_up_s = np.floor(self.n * p_s + self.c * np.sqrt(self.n * p_s *
                                                              (1 - p_s)) + 0.5)
            k_pos_s = self._k_bin(k_up_s)
            hst[ord, k_pos_s] += 1

        # Scores computation
        for i in range(s):
            count = 0
            k_pos = 0

            for l, rho in enumerate(self.rhos):
                while count < s * rho:
                    count += hst[i, k_pos]
                    k_pos += 1

                self.sc[start_i + i, l] = self._k_bin_inv(k_pos) / self.n

    def _k_bin(self, k_up):
        return (np.log(k_up) / self.binning_ratio_log).astype(int)

    def _k_bin_inv(self, k_pos):
        return self.binning_ratio**k_pos
