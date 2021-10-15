from typing import Callable, List, Optional, Union

import numpy as np
from scipy.spatial.distance import cdist


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

        rhos = np.array(rhos)
        if (rhos < 0.0).any() or (rhos > 1.0).any():
            raise ValueError(f'rhos ({rhos}) must be between 0 and 1')
        self.rhos = rhos

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

        self.log_spaced_bins = None
        self.n = None

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
        self.n, _ = X.shape
        self.log_spaced_bins = np.logspace(np.log10(1), np.log10(self.n),
                                           self.n_bins)
        self.sc = np.zeros((self.n, len(self.rhos)))
        self._fast_cfof(X)
        return self.sc

    def _fast_cfof(self, X: np.ndarray):
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
            part = X[a:b]
            self._fast_cfof_part(part, start_i=a)
            i = i + self.s

    def _fast_cfof_part(self, partition: np.ndarray, start_i: int):
        s, _ = partition.shape

        hst = np.zeros((s, self.n_bins))

        # Nearest neighbor count estimation
        for i in range(s):
            # Distances computation
            dst = cdist(partition[[i], :], partition, metric=self.metric)[0]

            # Count update
            ord = np.argsort(dst)

            for j in range(s):
                p = (j + 1) / s
                k_up = np.floor(self.n * p + self.c * np.sqrt(self.n * p *
                                                              (1 - p)) + 0.5)
                k_pos = self._k_bin(k_up)
                hst[ord[j], k_pos] += 1

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
        for i, b in enumerate(self.log_spaced_bins[1:]):
            if k_up < b:
                return i

    def _k_bin_inv(self, k_pos):
        # TODO: check this
        return self.log_spaced_bins[k_pos - 1]
