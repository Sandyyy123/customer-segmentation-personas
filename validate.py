"""
Cluster validation - beyond statistical separation.

Three layers, because a good silhouette alone proves nothing about whether a
segment is real or useful:

  1. separation      - silhouette (reported from cluster.py).
  2. stability        - bootstrap the data, re-cluster, and measure how often the
                        same customers land together (Adjusted Rand Index vs the
                        reference labelling). A segment that dissolves under
                        resampling is a mirage.
  3. external validity - does the segmentation predict a variable it never saw
                        (churn)? If churn rate differs across clusters far beyond
                        chance (chi-square), the segments carry real signal.
"""
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score


def bootstrap_stability(Z, reference_labels, k, n_boot=100, random_state=42):
    """
    Mean Adjusted Rand Index between the reference clustering and clusterings
    fit on bootstrap resamples. ~1.0 = the structure is rock solid; near 0 =
    noise. Reported per run so the client sees a number, not a promise.
    """
    rng = np.random.default_rng(random_state)
    n = Z.shape[0]
    aris = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)  # sample with replacement
        km = KMeans(n_clusters=k, n_init=5, random_state=int(rng.integers(0, 1e6)))
        boot_labels = km.fit_predict(Z[idx])
        aris.append(adjusted_rand_score(reference_labels[idx], boot_labels))
    aris = np.array(aris)
    return dict(mean_ari=round(float(aris.mean()), 3), std_ari=round(float(aris.std()), 3))


def external_validity(labels, external: pd.Series):
    """
    Chi-square test of independence between cluster membership and a held-out
    binary outcome (churn). A small p with a spread of per-cluster rates means
    the segments predict something they were never trained on.
    """
    tab = pd.crosstab(pd.Series(labels, name="cluster"), external)
    chi2, p, dof, _ = chi2_contingency(tab)
    rate = external.groupby(pd.Series(labels, index=external.index)).mean()
    return dict(
        chi2=round(float(chi2), 2),
        p_value=float(p),
        churn_rate_by_cluster={int(k): round(float(v), 3) for k, v in rate.items()},
    )
