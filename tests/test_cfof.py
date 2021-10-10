from cfof import CFOF

import numpy as np


class TestCFOF:
    def test_cfof(self):
        X = np.array([[9, 15], [64, 28], [89, 93], [29, 8], [73, 0], [40, 36],
                      [16, 11], [54, 88], [62, 33], [72, 78]])
        expected_scores = np.array(
            [0.7, 0.5, 0., 0.4, 0.7, 0.4, 0.6, 0.8, 0.4, 0.6])
        rho = 0.5
        clf = CFOF()
        output_scores = clf.cfof(X, rho=rho)
        assert (expected_scores == output_scores).all()
