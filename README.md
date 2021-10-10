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

## References

<a id="1">[1]</a>
ANGIULLI, Fabrizio. CFOF: a concentration free measure for anomaly detection. ACM Transactions on Knowledge Discovery from Data (TKDD), 2020, vol. 14, no 1, p. 1-53.
