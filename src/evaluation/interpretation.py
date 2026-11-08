"""Model interpretation: impurity importance, coefficients, permutation importance."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.preprocessing.pipeline import transformed_feature_names


def model_importance(pipeline: Pipeline) -> dict[str, Any] | None:
    """Model-intrinsic importance, if the estimator exposes one.

    * Trees/forests: mean decrease in impurity on *transformed* features
      (computed on training data; biased toward high-cardinality/continuous
      features).
    * Logistic regression: coefficients on standardised features (log-odds per
      1 SD change, or per category membership), plus odds ratios.
    * HistGradientBoosting and SVM have no comparable intrinsic importance -> ``None``.
    """
    clf = pipeline.named_steps["classifier"]
    names = transformed_feature_names(pipeline)
    # Only logistic-regression coefficients are log-odds. A linear-kernel SVM also has
    # `coef_`, but those weights have no probability meaning, so it gets no intrinsic view.
    if isinstance(clf, LogisticRegression):
        coef = clf.coef_.ravel()
        df = pd.DataFrame({"feature": names, "coefficient": coef,
                           "odds_ratio": np.exp(coef), "abs": np.abs(coef)})
        df = df.sort_values("abs", ascending=False).drop(columns="abs")
        return {"kind": "coefficients", "intercept": float(clf.intercept_[0]),
                "values": df.to_dict(orient="list")}
    if hasattr(clf, "feature_importances_"):
        df = pd.DataFrame({"feature": names, "importance": clf.feature_importances_})
        df = df.sort_values("importance", ascending=False)
        return {"kind": "impurity", "values": df.to_dict(orient="list")}
    return None


def permutation_importance_table(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series,
                                 n_repeats: int = 30, seed: int = 42,
                                 scoring: str = "roc_auc") -> dict[str, Any]:
    """Permutation importance of *raw input columns* on held-out data.

    Permuting a raw column also permutes everything derived from it (e.g. age
    drives `age`, `is_child` and the missing-age indicator), so one-hot dummies
    and engineered features are evaluated as a group.
    """
    result = permutation_importance(pipeline, X, y, n_repeats=n_repeats,
                                    random_state=seed, scoring=scoring, n_jobs=1)
    df = pd.DataFrame({"feature": X.columns, "mean": result.importances_mean,
                       "std": result.importances_std}).sort_values("mean", ascending=False)
    return {"scoring": scoring, "n_repeats": n_repeats, "values": df.to_dict(orient="list")}
