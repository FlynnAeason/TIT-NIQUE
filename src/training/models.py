"""Candidate models and their hyperparameter search spaces."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from scipy.stats import loguniform, randint, uniform
from sklearn.base import BaseEstimator
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


@dataclass(frozen=True)
class ModelSpec:
    """A candidate model: constructor, search space and a short rationale."""

    key: str
    display_name: str
    factory: Callable[[int], BaseEstimator]
    param_distributions: dict[str, Any]
    description: str
    tags: tuple[str, ...] = field(default_factory=tuple)


def _logistic(seed: int) -> BaseEstimator:
    return LogisticRegression(solver="liblinear", max_iter=2000, random_state=seed)


def _random_forest(seed: int) -> BaseEstimator:
    return RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=1)


def _decision_tree(seed: int) -> BaseEstimator:
    return DecisionTreeClassifier(random_state=seed)


def _hist_gb(seed: int) -> BaseEstimator:
    return HistGradientBoostingClassifier(random_state=seed, early_stopping=False)


def _svm(seed: int) -> BaseEstimator:
    # An SVM outputs distances, not probabilities. Platt scaling (a logistic fit on
    # cross-validated SVM scores, inside each training fold) turns them into
    # P(survived) like the other models. This is scikit-learn's recommended
    # replacement for the deprecated SVC(probability=True).
    return CalibratedClassifierCV(SVC(random_state=seed, cache_size=500), method="sigmoid",
                                  cv=5, ensemble=False)


P = "classifier__"
S = P + "estimator__"  # SVC parameters sit inside the calibration wrapper

MODEL_SPECS: dict[str, ModelSpec] = {
    "logistic_regression": ModelSpec(
        key="logistic_regression",
        display_name="Logistic Regression",
        factory=_logistic,
        param_distributions={
            P + "C": loguniform(1e-3, 1e2),
            # l1_ratio 0 = L2 (ridge), 1 = L1 (lasso); replaces the deprecated
            # `penalty` argument used in the notebook (sklearn >= 1.8).
            P + "l1_ratio": [0.0, 1.0],
            P + "class_weight": [None, "balanced"],
        },
        description="Regularised linear model on standardised features. Reference model "
                    "from the original notebook; the notebook did not tune C, which is "
                    "the most influential hyperparameter, so it is added here.",
        tags=("reference", "linear"),
    ),
    "random_forest": ModelSpec(
        key="random_forest",
        display_name="Random Forest",
        factory=_random_forest,
        param_distributions={
            P + "max_depth": [None, 4, 6, 8, 10, 14],
            P + "min_samples_split": randint(2, 21),
            P + "min_samples_leaf": randint(1, 11),
            P + "max_features": ["sqrt", "log2", 0.5, None],
            P + "class_weight": [None, "balanced"],
        },
        description="Bagged ensemble of decision trees (reference model from the notebook). "
                    "Search widened from the notebook grid to include leaf size and "
                    "feature subsampling, which control variance on small data.",
        tags=("reference", "ensemble"),
    ),
    "decision_tree": ModelSpec(
        key="decision_tree",
        display_name="Decision Tree",
        factory=_decision_tree,
        param_distributions={
            P + "max_depth": [2, 3, 4, 5, 6, 8, None],
            P + "min_samples_leaf": randint(1, 31),
            P + "criterion": ["gini", "entropy"],
            P + "class_weight": [None, "balanced"],
        },
        description="Single interpretable tree; a transparent baseline for the ensembles.",
        tags=("baseline", "interpretable"),
    ),
    "hist_gradient_boosting": ModelSpec(
        key="hist_gradient_boosting",
        display_name="Hist Gradient Boosting",
        factory=_hist_gb,
        param_distributions={
            P + "learning_rate": loguniform(0.01, 0.3),
            P + "max_iter": randint(50, 401),
            P + "max_depth": [None, 2, 3, 4, 6],
            P + "min_samples_leaf": randint(5, 51),
            P + "l2_regularization": uniform(0.0, 2.0),
        },
        description="Gradient-boosted trees (scikit-learn native, no extra dependency). "
                    "Typically the strongest tabular learner; tuned conservatively "
                    "because the training set is small.",
        tags=("ensemble", "boosting"),
    ),
    "svm": ModelSpec(
        key="svm",
        display_name="Support Vector Machine",
        factory=_svm,
        param_distributions={
            S + "kernel": ["rbf", "linear"],
            S + "C": loguniform(1e-4, 3e1),
            # gamma only affects the RBF kernel; it is ignored for "linear".
            S + "gamma": loguniform(1e-3, 1e0),
            S + "class_weight": [None, "balanced"],
        },
        description="Finds the boundary between survivors and non-survivors with the widest "
                    "margin; the RBF kernel allows a smooth curved boundary. Relies on the "
                    "pipeline's scaling. Probabilities come from internal Platt scaling. C is "
                    "capped at 30 to keep the linear kernel's training time bounded.",
        tags=("kernel",),
    ),
}


def get_model_specs(keys: list[str] | None = None) -> list[ModelSpec]:
    """Return model specs, optionally restricted to ``keys`` (order preserved)."""
    if keys is None:
        return list(MODEL_SPECS.values())
    unknown = [k for k in keys if k not in MODEL_SPECS]
    if unknown:
        raise KeyError(f"Unknown model keys {unknown}; available: {list(MODEL_SPECS)}")
    return [MODEL_SPECS[k] for k in keys]
