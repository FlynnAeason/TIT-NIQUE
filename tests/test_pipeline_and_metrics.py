"""Preprocessing leakage safety, split integrity and metric correctness."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

import config
from src.data.loader import split_features_target
from src.evaluation.metrics import bootstrap_ci, compute_metrics, confusion, threshold_table
from src.features.engineering import add_engineered_features
from src.preprocessing.pipeline import build_pipeline, build_preprocessor, transformed_feature_names
from src.training.train import split_data


def test_split_is_stratified_disjoint_and_reproducible(raw_df):
    X, y = split_features_target(raw_df)
    cfg = config.ExperimentConfig()
    X_tr, X_te, y_tr, y_te = split_data(X, y, cfg)
    assert set(X_tr.index).isdisjoint(X_te.index)
    assert len(X_tr) + len(X_te) == len(X)
    assert abs(y_tr.mean() - y_te.mean()) < 0.01
    X_tr2, X_te2, *_ = split_data(X, y, cfg)
    assert list(X_te.index) == list(X_te2.index)


def test_preprocessor_learns_only_from_fit_data(raw_df):
    """Imputation median must come from the data passed to fit, not from later data."""
    X, _ = split_features_target(raw_df)
    train, other = X.iloc[:400], X.iloc[400:].copy()
    other["age"] = 999.0  # extreme values that would shift a leaked median
    other.loc[other.index[:50], "age"] = np.nan
    pre = build_preprocessor().fit(train)
    imputer = pre.named_steps["columns"].named_transformers_["num"].named_steps["imputer"]
    assert imputer.statistics_[0] == pytest.approx(train["age"].median())
    pre.transform(other)  # transforming must not refit
    assert imputer.statistics_[0] == pytest.approx(train["age"].median())


def test_engineered_features():
    df = pd.DataFrame({"age": [5.0, 40.0, np.nan], "sibsp": [1, 0, 0], "parch": [2, 0, 0],
                       "fare": [0.0, np.e - 1, 10.0], "pclass": [3, 1, 2], "sex": ["male"] * 3,
                       "embarked": ["S", None, "C"]})
    out = add_engineered_features(df)
    assert out["is_child"].tolist() == [1.0, 0.0, 0.0]
    assert out["log_fare"].iloc[1] == pytest.approx(1.0)
    assert out["pclass"].iloc[0] == "3"
    assert pd.isna(out["embarked"].iloc[1])


def test_pipeline_output_has_no_nans_and_expected_names(raw_df):
    X, y = split_features_target(raw_df)
    pipe = build_pipeline(LogisticRegression(max_iter=1000)).fit(X, y)
    Z = pipe.named_steps["preprocessor"].transform(X)
    assert not np.isnan(Z).any()
    names = transformed_feature_names(pipe)
    assert "missingindicator_age" in names and "sex_female" in names and "is_child" in names
    assert not any(n.startswith(("alive", "who", "class_", "adult_male")) for n in names)


def test_pipeline_handles_unseen_category(raw_df):
    X, y = split_features_target(raw_df)
    pipe = build_pipeline(LogisticRegression(max_iter=1000)).fit(X[X["embarked"] != "Q"], y[X["embarked"] != "Q"])
    proba = pipe.predict_proba(X[X["embarked"] == "Q"].head(3))
    assert proba.shape == (3, 2)


def test_svm_spec_trains_and_gives_probabilities(raw_df):
    from src.training.models import get_model_specs
    spec = get_model_specs(["svm"])[0]
    X, y = split_features_target(raw_df)
    pipe = build_pipeline(spec.factory(0)).set_params(classifier__estimator__kernel="rbf", classifier__estimator__C=1.0)
    proba = pipe.fit(X.iloc[:400], y.iloc[:400]).predict_proba(X.iloc[400:450])
    assert proba.shape == (50, 2) and ((proba >= 0) & (proba <= 1)).all()


def test_linear_svm_is_not_reported_as_log_odds(raw_df):
    from src.evaluation.interpretation import model_importance
    from src.training.models import get_model_specs
    X, y = split_features_target(raw_df)
    pipe = build_pipeline(get_model_specs(["svm"])[0].factory(0)).set_params(classifier__estimator__kernel="linear")
    assert model_importance(pipe.fit(X, y)) is None


def test_compute_metrics_against_sklearn():
    y = np.array([0, 0, 1, 1, 1, 0])
    p = np.array([0.1, 0.6, 0.7, 0.4, 0.9, 0.2])
    m = compute_metrics(y, p)
    assert m["accuracy"] == pytest.approx(accuracy_score(y, p >= 0.5))
    assert m["roc_auc"] == pytest.approx(roc_auc_score(y, p))
    # TP=2 FP=1 FN=1 TN=2
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == pytest.approx(2 / 3)
    assert m["specificity"] == pytest.approx(2 / 3)
    assert confusion(y, (p >= 0.5).astype(int)) == {"tn": 2, "fp": 1, "fn": 1, "tp": 2}


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_fast_metrics_match_reference(seed):
    from src.evaluation.metrics import fast_metrics
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, 150)
    p = rng.random(150)
    ref, fast = compute_metrics(y, p), fast_metrics(y, p)
    for k, v in fast.items():
        assert v == pytest.approx(ref[k]), k


def test_undefined_metrics_are_nan_not_zero():
    m = compute_metrics(np.array([1, 1, 1]), np.array([0.2, 0.3, 0.4]))
    assert np.isnan(m["roc_auc"]) and np.isnan(m["precision"]) and np.isnan(m["mcc"])
    assert m["recall"] == 0.0


def test_bootstrap_interval_contains_estimate():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    p = np.clip(y * 0.6 + rng.normal(0.2, 0.2, 200), 0, 1)
    ci = bootstrap_ci(y, p, n_boot=200, seed=1)
    for k in ("roc_auc", "accuracy", "f1"):
        assert ci[k]["low"] <= ci[k]["estimate"] <= ci[k]["high"]


def test_threshold_table_monotone_recall():
    y = np.array([0, 1, 0, 1, 1, 0, 1, 0])
    p = np.linspace(0.05, 0.95, 8)
    recalls = [r["recall"] for r in threshold_table(y, p)]
    assert all(a >= b for a, b in zip(recalls, recalls[1:]))
