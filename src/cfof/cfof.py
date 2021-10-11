from typing import Callable, List, Union
from multiprocessing import Pool
from functools import partial

import numpy as np
from sklearn.neighbors import NearestNeighbors


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

        self.log_spaced_bins = None

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

        # `neighbors[i]` are the neighbours of i in order of proximity.
        # Every point is the first neighbor of itself.
        neighbors = self._find_neighbors(X)

        # `min_k_neighborhood[i, j]` represents min k such that i contains j
        # in its neighborhood.
        min_k_neighborhood = np.argsort(neighbors) + 1

        threshold = rho * n

        with Pool(processes=self.n_jobs) as pool:
            cfof_scores = pool.map(
                partial(CFOF._compute_col_cfof, threshold=threshold, n=n),
                min_k_neighborhood.T)

        return np.array(cfof_scores)

    def _find_neighbors(self, X: np.ndarray, algorithm='auto'):
        nbrs = NearestNeighbors(n_neighbors=len(X),
                                algorithm=algorithm,
                                metric=self.metric,
                                n_jobs=self.n_jobs).fit(X)
        indices = nbrs.kneighbors(X, return_distance=False)
        return indices

    def _compute_col_cfof(col, threshold, n):
        counter = np.bincount(col).cumsum()
        return np.argmax(counter >= threshold) / n

    # TODO: use faiss / CPU / GPU
    # if use_faiss:
    #     import faiss
    #     faiss_index = faiss.IndexFlatL2(X.shape[1])
    #     faiss_index.add(X.astype(np.float32))
    #     # Squared euclidean
    #     _, neighbors = faiss_index.search(X.astype(np.float32), k=n)

    def fast_cfof(self,
                  X: np.ndarray,
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
        self.log_spaced_bins = np.logspace(np.log10(1), np.log10(n), b)

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

    def _fast_cfof_part(self, partition: np.ndarray, rhos: List[float], b: int,
                        n: int):
        s, _ = partition.shape
        pass

    def _k_bin(self, k_up):
        return np.argmax(self.log_spaced_bins >= k_up)

    def _k_bin_inv(self):
        pass
