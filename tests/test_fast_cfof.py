import numpy as np

from cfof import FastCFOF


class TestFastCFOF:
    def test_fast_cfof_simple(self):
        expected_score_0_0 = 0.00954095

        np.random.seed(10)
        X = np.random.randint(0, 100, size=(1000, 3))

        fast_cfof_clf = FastCFOF(metric='euclidean',
                                 rhos=[0.001, 0.005, 0.01, 0.05, 0.1],
                                 epsilon=0.1,
                                 delta=0.1,
                                 n_bins=50,
                                 n_jobs=1)

        scores = fast_cfof_clf.compute(X)

        assert np.isclose(scores[0][0], expected_score_0_0)
