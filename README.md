> **⚠️ Proprietary — All Rights Reserved.** © 2026 Sandeep Grover. This repository is licensed to Sandeep Grover and may **not** be used, run, copied, modified, distributed, or used to train models without prior written permission. Public visibility does not grant a license. See [LICENSE](LICENSE).

---

# Customer Segmentation & Persona Analysis (demo)

A runnable, self-contained demo of a three-phase B2B customer segmentation
engagement: **demographic personas**, **behavioral segments**, and the
**synthesis** that asks whether who a customer *is* predicts how they *behave*.

It is built around the question a serious buyer always asks:

> *how did you check the clusters held up **beyond statistical separation**?*

The answer here is three concrete layers, not a single silhouette score.

---

## What it does

```
data.py      synthetic B2B accounts (mixed categorical + continuous) +
             in-app usage + a HELD-OUT churn flag never used for clustering
features.py  mixed-type feature engineering: log-transform skewed continuous,
             standardise, one-hot categoricals  (ColumnTransformer)
cluster.py   PCA (90%+ variance) -> K-means, k chosen by silhouette over 2..7
validate.py  1) silhouette          - statistical separation
             2) bootstrap stability - ARI across 100 resamples
             3) external validity   - chi-square of clusters vs held-out churn
personas.py  cluster -> named, sized, human-readable persona card
main.py      runs Phase 1 + validation + a Phase 3 synthesis crosstab
```

## Architecture

```
   customers.csv            usage.csv
        |                       |
        v                       |
  feature engineering           |
  (mixed types)                 |
        |                       |
        v                       |
      PCA                       |
        |                       |
        v                       |
     K-means  --choose k by silhouette                     Phase 2
        |                                                (behavioral
        +--> silhouette / gap        (separation)         segments)
        +--> bootstrap ARI           (stability)              |
        +--> chi-square vs churn     (external validity) <----+
        |                                                     |
        v                                                     v
     personas  ------------ Phase 3 synthesis: persona x behavior crosstab
```

## Run it

```bash
pip install -r requirements.txt
python main.py
```

Example output (synthetic seed 42):

```
Chosen k                   : 4  (silhouette 0.3963)
Bootstrap stability (ARI)  : 1.0 +/- 0.0  (100 resamples)
External validity vs churn : chi2=108.83, p=1.96e-23
Validity verdict           : PASS (stable under resampling AND predicts held-out churn)
```

The clusters are recovered at ARI 0.998 against the known latent groups, are
stable under resampling, and predict a churn signal they were never shown - the
combination that makes a segmentation safe to put a budget behind.

## Why these choices

- **Mixed data is handled explicitly.** Continuous income and categorical
  business type do not belong on the same scale; forcing them through one
  encoder is where most segmentations quietly break. For a heavier categorical
  mix, swap in k-prototypes / Gower distance (noted in `cluster.py`).
- **k is measured, not guessed** - silhouette over a candidate range.
- **Validation is the point.** Separation, stability, and external validity are
  three different questions. A cluster can be tight and still be noise.

## Notes

- All data is synthetic; no client data is required to run this.
- This is a demonstration of method and code quality, not a finished client
  deliverable.

Dr. Sandeep Grover - Python / pandas / scikit-learn, PCA & clustering on
high-dimensional mixed data.
