"""
Mixed-type feature engineering.

The single most common way a segmentation goes wrong is treating a categorical
field (business_type) and a continuous field (annual_income) as if they live on
the same scale. Here continuous columns are log-transformed where skewed and
standardised; categoricals are one-hot encoded. The result is a numeric matrix
suitable for PCA + K-means.

For a heavier categorical mix, prefer k-prototypes or a Gower distance matrix -
see cluster.py for where that swap would go.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer

CONTINUOUS = ["annual_income", "account_holder_age", "seats"]
CATEGORICAL = ["business_type", "firm_size", "region"]
SKEWED = ["annual_income", "seats"]  # long right tails -> log1p before scaling


def _log_skewed(X):
    X = np.asarray(X, dtype=float)
    return np.log1p(X)


def build_preprocessor() -> ColumnTransformer:
    cont_pipe = Pipeline(
        steps=[
            ("log", FunctionTransformer(_log_skewed, feature_names_out="one-to-one")),
            ("scale", StandardScaler()),
        ]
    )
    # non-skewed continuous just get scaled
    plain_cont = [c for c in CONTINUOUS if c not in SKEWED]
    return ColumnTransformer(
        transformers=[
            ("skewed", cont_pipe, SKEWED),
            ("cont", StandardScaler(), plain_cont),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
        ],
        remainder="drop",
    )


def transform(customers: pd.DataFrame):
    pre = build_preprocessor()
    X = pre.fit_transform(customers[CONTINUOUS + CATEGORICAL])
    names = pre.get_feature_names_out()
    return X, names, pre
