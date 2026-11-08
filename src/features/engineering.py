"""Row-wise feature engineering.

These transforms use no statistics learned from data (no means, no quantiles),
so applying them inside the pipeline cannot leak information between folds.
They live in a module-level function so fitted pipelines can be pickled.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config

ENGINEERED_NUMERIC: list[str] = ["age", "sibsp", "parch", "log_fare"]
ENGINEERED_BINARY: list[str] = ["is_child"]
ENGINEERED_CATEGORICAL: list[str] = list(config.CATEGORICAL_FEATURES)

FEATURE_DESCRIPTIONS: dict[str, str] = {
    "age": "Age in years (median-imputed inside each training fold; a "
           "missing-age indicator is added).",
    "sibsp": "Number of siblings/spouses aboard.",
    "parch": "Number of parents/children aboard.",
    "log_fare": "log(1 + fare); reduces the strong right skew of fares.",
    "is_child": f"1 if age < {config.CHILD_AGE_THRESHOLD:g}, else 0 (0 when age is "
                "unknown). Captures the 'women and children first' threshold that "
                "the dropped `who` column encoded.",
    "pclass": "Ticket class (1/2/3), one-hot encoded.",
    "sex": "Sex, one-hot encoded.",
    "embarked": "Port of embarkation (C/Q/S), most-frequent-imputed and one-hot encoded.",
}


def add_engineered_features(X: pd.DataFrame) -> pd.DataFrame:
    """Return a frame with the columns expected by the column transformer."""
    out = pd.DataFrame(index=X.index)
    out["age"] = X["age"].astype(float)
    out["sibsp"] = X["sibsp"].astype(float)
    out["parch"] = X["parch"].astype(float)
    out["log_fare"] = np.log1p(X["fare"].astype(float))
    out["is_child"] = (X["age"].astype(float) < config.CHILD_AGE_THRESHOLD).astype(float)
    for col in ENGINEERED_CATEGORICAL:
        # Cast to string so ints (pclass) and strings share one representation.
        out[col] = X[col].astype("object").where(X[col].notna(), np.nan).map(
            lambda v: v if (isinstance(v, float) and np.isnan(v)) else str(v)
        ).astype("object")
    return out
