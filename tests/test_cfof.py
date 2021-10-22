import numpy as np

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
