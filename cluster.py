"""
Dimensionality reduction + clustering.

PCA first (keep enough components for ~90% variance, and to de-correlate the
one-hot columns), then K-means. k is chosen by silhouette over a candidate
range - not eyeballed. Returns the fitted objects so the same transform can be
replayed on new customers next quarter.
"""
from dataclasses import dataclass
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


@dataclass
class ClusterResult:
    labels: np.ndarray
    k: int
    pca: PCA
    kmeans: KMeans
    silhouette: float
    scores_by_k: dict


def choose_k_and_fit(X, k_range=range(2, 8), variance=0.90, random_state=42) -> ClusterResult:
    pca = PCA(n_components=variance, random_state=random_state)
    Z = pca.fit_transform(X)

    scores = {}
    best = None
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
        labels = km.fit_predict(Z)
        sil = silhouette_score(Z, labels)
        scores[k] = round(float(sil), 4)
        if best is None or sil > best[0]:
            best = (sil, k, km, labels)

    sil, k, km, labels = best
    return ClusterResult(
        labels=labels,
        k=k,
        pca=pca,
        kmeans=km,
        silhouette=round(float(sil), 4),
        scores_by_k=scores,
    )
