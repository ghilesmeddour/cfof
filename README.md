# CFOF (Concentration Free Outlier Factor)

🚧 **Work In Progress..**

Python implementation of Concentration Free Outlier Factor (CFOF) [[1]](#1).

## CFOF properties

- [**Concentration**](## "The tendency of distances to become almost indiscernible as dimensionality
increases.") free
- Does not suffer of the **hubness** problem
- [**Semi–locality**](## "CFOF score is both translation and scale-invariant and, hence, that the number of outliers coming from each cluster is directly proportional to its size and to its kurtosis")
- **fast-CFOF** algorithm allows to calculate reliably CFOF scores with linear cost both in the dataset size and dimensionality

## Installation

To install the latest release:
```
$ pip install cfof
```

## Usage

Import `CFOF` and `FastCFOF`.

```python
>>> from cfof import CFOF, FastCFOF
>>> import numpy as np
```

Load data.

```python
>>> X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
```

Instantiate `CFOF` or `FastCFOF`, then call `.compute(X)` to calculate the scores. `.compute(X)` returns `sc`, where `sc[i, l]` is score of object `i` for `ϱ_l` (rhos[l]).

You can also calculate CFOF scores from a precomputed distance matrix using 
`.compute_from_distance_matrix()`.

### CFOF (hard-CFOF)

Use `compute` to compute CFOF scores directly from data.

```python
>>> cfof_clf = CFOF(metric='euclidean', rhos=[0.5, 0.6], n_jobs=1)
>>> cfof_clf.compute(X)
array([[0.5       , 0.66666667],
       [0.33333333, 0.83333333],
       [0.5       , 1.        ],
       [0.5       , 0.66666667],
       [0.33333333, 0.83333333],
       [0.5       , 1.        ]])
```

Use `compute_from_distance_matrix` to compute CFOF scores from a precomputed 
distance matrix.

```python
>>> from sklearn.metrics import pairwise_distances
>>> distance_matrix = pairwise_distances(X, metric='euclidean')
>>> cfof_clf.compute_from_distance_matrix(distance_matrix)
array([[0.5       , 0.66666667],
       [0.33333333, 0.83333333],
       [0.5       , 1.        ],
       [0.5       , 0.66666667],
       [0.33333333, 0.83333333],
       [0.5       , 1.        ]])
```

### FastCFOF (soft-CFOF)

Use `compute` to compute CFOF scores directly from data.

```python
>>> np.random.seed(10)
>>> X = np.random.randint(0, 100, size=(1000, 3))
>>>
>>> fast_cfof_clf = FastCFOF(metric='euclidean',
...                          rhos=[0.001, 0.005, 0.01, 0.05, 0.1],
...                          epsilon=0.1, delta=0.1, n_bins=50, n_jobs=1)
>>> fast_cfof_clf.compute(X)
array([[0.00954095, 0.00954095, 0.01930698, 0.05963623, 0.10481131],
       [0.00954095, 0.00954095, 0.01930698, 0.06866488, 0.10481131],
       [0.00954095, 0.00954095, 0.02559548, 0.06866488, 0.10481131],
       ...,
       [0.00954095, 0.00954095, 0.01930698, 0.05963623, 0.10481131],
       [0.00954095, 0.00954095, 0.03393222, 0.15998587, 0.24420531],
       [0.00954095, 0.00954095, 0.02559548, 0.0390694 , 0.09102982]])
```

Use `compute_from_distance_matrix` to compute CFOF scores from a precomputed 
distance matrix.

```python
>>> from sklearn.metrics import pairwise_distances
>>> distance_matrix = pairwise_distances(X, metric='euclidean')
>>> fast_cfof_clf.compute_from_distance_matrix(distance_matrix)
array([[0.00954095, 0.00954095, 0.01930698, 0.05963623, 0.10481131],
       [0.00954095, 0.00954095, 0.01930698, 0.06866488, 0.10481131],
       [0.00954095, 0.00954095, 0.02559548, 0.06866488, 0.10481131],
       ...,
       [0.00954095, 0.00954095, 0.01930698, 0.05963623, 0.10481131],
       [0.00954095, 0.00954095, 0.03393222, 0.15998587, 0.24420531],
       [0.00954095, 0.00954095, 0.02559548, 0.0390694 , 0.09102982]])
```

### CFOFiSAX

This library provides a wrapper for [pyCFOFiSAX](https://github.com/luk-f/pyCFOFiSAX) [[2]](#2)

```python
>>> from cfof.cfof_isax import CFOFiSAXWrapper
```

Refer to [`pyCFOFiSAX` documentation](https://pycfofisax.readthedocs.io/fr/main/) 
for more details.

### TODOs

- [ ] Add support for [`faiss`](https://github.com/facebookresearch/faiss) (GPU).
- [ ] Parallelize FastCFOF.
- [ ] Add unit tests.
- [ ] Add benchmarks.
- [ ] Wrap [pyCFOFiSAX](https://github.com/luk-f/pyCFOFiSAX).

## References

<a id="1">[1]</a>
ANGIULLI, Fabrizio. CFOF: a concentration free measure for anomaly detection. ACM Transactions on Knowledge Discovery from Data (TKDD), 2020, vol. 14, no 1, p. 1-53.

<a id="2">[2]</a>
FOULON, Lucas, FENET, Serge, RIGOTTI, Christophe, et al. Scoring Message Stream Anomalies in Railway Communication Systems. In : 2019 International Conference on Data Mining Workshops (ICDMW). IEEE, 2019. p. 769-776.
