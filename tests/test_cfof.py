import numpy as np
from sklearn.metrics import pairwise_distances

from cfof import CFOF


class TestCFOF:
    def test_cfof_simple(self):
        expected_scores = np.array([[0.5,
                                     0.66666667], [0.33333333, 0.83333333],
                                    [0.5, 1.], [0.5, 0.66666667],
                                    [0.33333333, 0.83333333], [0.5, 1.]])

        X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
        cfof_clf = CFOF(metric='euclidean', rhos=[0.5, 0.6], n_jobs=1)
        scores = cfof_clf.compute(X)

        assert np.allclose(expected_scores, scores)

    def test_cfof_compute_and_compute_from_same_result(self):
        X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])

        cfof_clf = CFOF(metric='euclidean', rhos=[0.5, 0.6], n_jobs=1)

        scores = cfof_clf.compute(X)

        distance_matrix = pairwise_distances(X, metric='euclidean')
        scores_from_distance_matrix = cfof_clf.compute_from_distance_matrix(
            distance_matrix)

        assert np.allclose(scores_from_distance_matrix, scores)
