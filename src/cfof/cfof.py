from typing import Callable, List, Union

import numpy as np
from sklearn.metrics import pairwise_distances


class CFOF:
    """
    Concentration Free Outlier Factor.

    Parameters
    ----------
    metric : str or callable, default 'euclidean'
        Must be as expected by `sklearn.metrics.pairwise_distances`.
    n_jobs : int, default None
        The number of jobs to use for the computation.
        `-1` means using all processors.
    """
    def __init__(self,
                 metric: Union[str, Callable] = 'euclidean',
                 n_jobs=None) -> None:
        self.metric = metric
        self.n_jobs = n_jobs

    def cfof(self, X: np.ndarray, rho: float = 0.01) -> np.ndarray:
        """
        Lorem.

        Parameters
        ----------
        X : numpy.ndarray
            Dataset.
        rho : float, default 0.01
            ϱ parameter, fraction of the data population.
            Must be between 0 and 1.
        """
        if not (0.0 <= rho <= 1.0):
            raise ValueError(f'rho ({rho}) must be between 0 and 1')

        n, _ = X.shape

        # `pairwise_distances[i, j]` is the distance between i and j.
        pd_matrix = pairwise_distances(X,
                                       metric=self.metric,
                                       n_jobs=self.n_jobs)

        # `neighbors[i]` are the neighbours of i in order of proximity.
        # Every point is the first neighbor of itself.
        neighbors = np.argsort(pd_matrix)

        # `min_k_neighborhood[i, j]` represents min k such that i contains j
        # in its neighborhood.
        # TODO: add 1 ?
        min_k_neighborhood = np.argsort(neighbors)

        threshold = rho * n

        cfof_scores = np.zeros((n, ))
        for i in range(n):
            for k in range(1, n):
                if (min_k_neighborhood[:, i] <= k).sum() >= threshold:
                    cfof_scores[i] = k / n
                    break

        return cfof_scores

    @staticmethod
    def fast_cfof(X: np.ndarray,
                  rhos: List[float] = [0.001, 0.005, 0.01, 0.05, 0.1],
                  epsilon: float = 0.01,
                  delta: float = 0.01,
                  b: int = 10) -> np.ndarray:
        """
        Lorem.

        Parameters
        ----------
        X : numpy.ndarray
            Dataset.
        rhos : List[float], default [0.001, 0.005, 0.01, 0.05, 0.1]
            ϱ parameter, fraction of the data population.
            Must be between 0 and 1.
        epsilon : float, default 0.01
            Lorem.
        delta : float, default 0.01
            Lorem.
        b : int, default 10
            Histogram bins.
        """
        n, _ = X.shape

        # The size s of the sample (or partition) of the dataset needed
        s = int(np.ceil((1 / 2 * (epsilon**2)) * np.log(2 / delta)))
        i = 0

        while i < n:
            if i + s < n:
                a = i + 1
            else:
                a = n - s + 1
            b = a + s - 1
            part = X[a:b]
            _ = CFOF._fast_cfof_part(part, rhos, b, n)
            i = i + s

        return None

    @staticmethod
    def _fast_cfof_part(partition: np.ndarray, rhos: List[float], b: int,
                        n: int):
        s, _ = partition.shape
        pass

    @staticmethod
    def _k_bin():
        pass
