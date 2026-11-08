"""Shared fixtures. Tests use the real dataset and a small, fast experiment."""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config  # noqa: E402


@pytest.fixture(scope="session")
def raw_df():
    from src.data.loader import load_raw_data
    if not config.RAW_DATA_FILE.exists():
        pytest.skip("Dataset not downloaded; run `python -m src.data.loader`.")
    return load_raw_data()


@pytest.fixture(scope="session")
def fast_cfg():
    return replace(config.ExperimentConfig(), nested_cv=False, n_iter=2, n_bootstrap=50,
                   permutation_repeats=2, cv_folds=3, n_jobs=1)


@pytest.fixture(scope="session")
def trained(tmp_path_factory, raw_df, fast_cfg):
    """Run a real (tiny) experiment into a temporary directory."""
    from src.training.train import run_training
    base = tmp_path_factory.mktemp("experiment")
    dirs = {"models_dir": base / "models", "reports_dir": base / "reports",
            "processed_dir": base / "processed"}
    results = run_training(fast_cfg, model_keys=["logistic_regression", "decision_tree"], **dirs)
    return results, dirs


@pytest.fixture
def passenger():
    return {"pclass": 1, "sex": "female", "age": 29.0, "sibsp": 0, "parch": 0,
            "fare": 100.0, "embarked": "S"}
