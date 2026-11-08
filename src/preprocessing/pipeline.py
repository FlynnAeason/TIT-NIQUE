"""Preprocessing pipeline construction.

All learned preprocessing (imputation medians, scaling statistics, encoder
categories) lives inside an sklearn ``Pipeline`` so it is re-fitted on the
training portion of every cross-validation fold and never sees validation or
test rows.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src.features.engineering import (
    ENGINEERED_BINARY,
    ENGINEERED_CATEGORICAL,
    ENGINEERED_NUMERIC,
    add_engineered_features,
)


def build_preprocessor() -> Pipeline:
    """Feature engineering followed by type-specific imputation/encoding."""
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent", missing_values=np.nan)),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    columns = ColumnTransformer(
        transformers=[
            ("num", numeric, ENGINEERED_NUMERIC),
            ("bin", "passthrough", ENGINEERED_BINARY),
            ("cat", categorical, ENGINEERED_CATEGORICAL),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return Pipeline([
        ("engineer", FunctionTransformer(add_engineered_features, validate=False)),
        ("columns", columns),
    ])


def build_pipeline(estimator: BaseEstimator) -> Pipeline:
    """Full model pipeline: preprocessing + classifier."""
    return Pipeline([("preprocessor", build_preprocessor()), ("classifier", estimator)])


def transformed_feature_names(pipeline: Pipeline) -> list[str]:
    """Names of the columns produced by a *fitted* pipeline's preprocessor."""
    return list(pipeline.named_steps["preprocessor"].named_steps["columns"].get_feature_names_out())
