"""Data loading, validation, profiling and feature selection."""

import numpy as np
import pandas as pd
import pytest

import config
from src.data.loader import DatasetNotFoundError, load_raw_data, split_features_target
from src.data.profiling import redundancy_checks, survival_rate_by, wilson_interval
from src.data.validation import DataValidationError, validate_features, validate_raw_dataset
from src.utils import file_sha256


def test_dataset_matches_pinned_hash():
    assert file_sha256(config.RAW_DATA_FILE) == config.DATA_SHA256


def test_load_shape_and_target(raw_df):
    assert raw_df.shape == (891, 15)
    assert set(raw_df[config.TARGET].unique()) == {0, 1}


def test_missing_file_raises(tmp_path):
    with pytest.raises(DatasetNotFoundError):
        load_raw_data(tmp_path / "nope.csv")


def test_raw_validation_rejects_missing_columns(raw_df):
    with pytest.raises(DataValidationError, match="Missing expected columns"):
        validate_raw_dataset(raw_df.drop(columns=["fare"]))


def test_raw_validation_rejects_non_binary_target(raw_df):
    bad = raw_df.copy()
    bad.loc[0, config.TARGET] = 2
    with pytest.raises(DataValidationError, match="binary"):
        validate_raw_dataset(bad)


def test_features_exclude_leaking_and_redundant_columns(raw_df):
    X, y = split_features_target(raw_df)
    assert list(X.columns) == config.MODEL_FEATURES
    for col in ("alive", "survived", "class", "who", "adult_male", "alone", "embark_town", "deck"):
        assert col not in X.columns
    assert y.name == config.TARGET


def test_redundancy_claims_hold_in_data(raw_df):
    checks = {c["column"]: c for c in redundancy_checks(raw_df)}
    for col in ("alive", "class", "who", "adult_male", "alone", "embark_town"):
        assert checks[col]["holds"] is True, col
        assert col in config.EXCLUDED_FEATURES


def test_validate_features_coerces_types():
    df = pd.DataFrame({"pclass": ["1"], "sex": [" Female "], "age": ["22"], "sibsp": [1],
                       "parch": [0], "fare": ["7.25"], "embarked": ["s"]})
    out = validate_features(df)
    assert out.loc[0, "pclass"] == 1 and out.loc[0, "sex"] == "female"
    assert out.loc[0, "embarked"] == "S" and out.loc[0, "age"] == 22.0


def test_validate_features_allows_imputable_missing():
    df = pd.DataFrame({"pclass": [3], "sex": ["male"], "age": [None], "sibsp": [0],
                       "parch": [0], "fare": [np.nan], "embarked": [None]})
    out = validate_features(df)
    assert out[["age", "fare", "embarked"]].isna().all(axis=None)


def test_validate_features_reports_all_errors():
    df = pd.DataFrame({"pclass": [4], "sex": ["robot"], "age": [-3], "sibsp": [1.5],
                       "parch": [0], "fare": ["abc"], "embarked": ["X"]})
    with pytest.raises(DataValidationError) as exc:
        validate_features(df)
    msg = " ".join(exc.value.errors)
    for col in ("pclass", "sex", "age", "sibsp", "fare", "embarked"):
        assert f"'{col}'" in msg


def test_validate_features_requires_columns():
    with pytest.raises(DataValidationError, match="Missing required columns"):
        validate_features(pd.DataFrame({"pclass": [1]}))


def test_validate_features_requires_sex():
    df = pd.DataFrame({"pclass": [1], "sex": [None], "age": [3], "sibsp": [0], "parch": [0],
                       "fare": [5], "embarked": ["S"]})
    with pytest.raises(DataValidationError, match="'sex' is required"):
        validate_features(df)


def test_survival_rate_by_handles_missing_category(raw_df):
    t = survival_rate_by(raw_df, "embarked")
    assert "missing" in set(t["embarked"])
    assert t["count"].sum() == len(raw_df)


def test_wilson_interval_bounds():
    lo, hi = wilson_interval(5, 10)
    assert 0 < lo < 0.5 < hi < 1
