from typing import List

import numpy as np
from pyCFOFiSAX import CFOFiSAX


# TODO: improve and check this.
class CFOFiSAXWrapper:
    """
    CFOFiSAX wrapper.

    Parameters
    ----------
    reference_data : numpy.array
        Refrence data.
    rhos : List[float], default [0.01]
        `ϱ` parameters, fraction of the data population.
        Must be between 0 and 1.
    verbose : bool, default False
        Verbosity.

    Notes
    -----
        Refer to `pyCFOFiSAX` 
        `documentation <https://pycfofisax.readthedocs.io/fr/main/>`_ 
        for more details.

    References
    ----------
    .. [2] FOULON, Lucas, FENET, Serge, RIGOTTI, Christophe, et al. 
           Scoring Message Stream Anomalies in Railway Communication Systems. 
           In : 2019 International Conference on Data Mining Workshops (ICDMW). 
           IEEE, 2019. p. 769-776.
    """
    def __init__(self,
                 reference_data: np.array,
                 rhos: List[float] = [0.001, 0.005, 0.01, 0.05, 0.1],
                 size_word: int = 200,
                 threshold: int = 30,
                 base_cardinality: int = 2,
                 number_tree: int = 20,
                 verbose=False) -> None:

        self.reference_data = reference_data
        self.rhos = rhos

        self.cfof_isax = CFOFiSAX()
        self.cfof_isax.init_forest_isax(data_ts=self.reference_data,
                                        size_word=size_word,
                                        threshold=threshold,
                                        base_cardinality=base_cardinality,
                                        number_tree=number_tree)
        self.cfof_isax.forest_isax.index_data(self.reference_data)
        self.cfof_isax.forest_isax.preprocessing_forest_for_icfof(
            self.reference_data, bool_print=verbose, count_num_node=verbose)

    def compute(self, X: np.ndarray) -> np.ndarray:
        """
        Compute iSAX CFOF scores.

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
        sc = self.cfof_isax.score_icfof(X,
                                        self.reference_data,
                                        rho=self.rhos,
                                        each_tree_score=True,
                                        fast_method=True)
        return sc
