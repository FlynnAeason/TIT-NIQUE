"""Cached artifact loaders and empty-state helpers for the Streamlit dashboard.

Caches are keyed on file modification times, so re-training refreshes the
dashboard automatically without a server restart.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

import config
from src.data.loader import DatasetNotFoundError, load_raw_data
from src.prediction.predict import LoadedModel, ModelUnavailableError, load_model
from src.utils import read_json


def _mtime(path: Path) -> float:
    return path.stat().st_mtime if path.exists() else 0.0


@st.cache_data(show_spinner=False, max_entries=2)
def _load_data(_mtime_key: float) -> pd.DataFrame:
    return load_raw_data()


@st.cache_data(show_spinner=False, max_entries=2)
def _load_results(_mtime_key: float) -> dict[str, Any]:
    return read_json(config.RESULTS_FILE)


@st.cache_data(show_spinner=False, max_entries=2)
def _load_predictions(_mtime_key: float) -> pd.DataFrame:
    return pd.read_csv(config.TEST_PREDICTIONS_FILE)


@st.cache_resource(show_spinner=False, max_entries=2)
def _load_model(_mtime_key: float) -> LoadedModel:
    return load_model()


MULTI_SEED_FILE = config.REPORTS_DIR / "multi_seed_results.json"


@st.cache_data(show_spinner=False, max_entries=2)
def _load_multi_seed(_mtime_key: float) -> dict[str, Any]:
    return read_json(MULTI_SEED_FILE)


def multi_seed() -> dict[str, Any] | None:
    """Results of `python -m src.training.multi_seed`, or None if it has not been run."""
    if not MULTI_SEED_FILE.exists():
        return None
    return _load_multi_seed(_mtime(MULTI_SEED_FILE))


SPLIT_FILE = config.PROCESSED_DATA_DIR / "split_indices.csv"


@st.cache_data(show_spinner=False, max_entries=2)
def _ladder(_mtime_key: tuple[float, float]) -> list[dict[str, Any]] | None:
    from src.evaluation.baselines import baseline_ladder
    res, df = results(), data()
    if res is None or df is None or not SPLIT_FILE.exists():
        return None
    split = pd.read_csv(SPLIT_FILE)
    sel = res["models"][res["selection"]["selected_model"]]
    return baseline_ladder(df, pd.Index(split.loc[split["split"] == "test", "row_index"]),
                           sel["display_name"], sel["test_metrics"])


def ladder() -> list[dict[str, Any]] | None:
    """Baseline ladder for the selected model on the saved test split (None if unavailable)."""
    return _ladder((_mtime(config.RESULTS_FILE), _mtime(SPLIT_FILE)))


@st.cache_resource(show_spinner=False, max_entries=2)
def _all_models(_mtime_key: float) -> dict[str, LoadedModel]:
    res = results()
    if res is None:
        return {}
    out = {}
    for key, m in res["models"].items():
        try:
            lm = load_model(m["model_file"])
        except ModelUnavailableError:
            continue
        lm.display_name = m["display_name"]
        out[key] = lm
    return out


def all_models() -> dict[str, LoadedModel]:
    """Every trained model that passes the hash check, keyed by model key."""
    return _all_models(_mtime(config.METADATA_FILE))


def data() -> pd.DataFrame | None:
    try:
        return _load_data(_mtime(config.RAW_DATA_FILE))
    except DatasetNotFoundError:
        return None


def results() -> dict[str, Any] | None:
    if not config.RESULTS_FILE.exists():
        return None
    return _load_results(_mtime(config.RESULTS_FILE))


def predictions() -> pd.DataFrame | None:
    if not config.TEST_PREDICTIONS_FILE.exists():
        return None
    return _load_predictions(_mtime(config.TEST_PREDICTIONS_FILE))


def model() -> LoadedModel | None:
    try:
        return _load_model(_mtime(config.BEST_MODEL_FILE))
    except ModelUnavailableError:
        return None


def require_data() -> pd.DataFrame:
    df = data()
    if df is None:
        st.warning(f"The dataset was not found at `{config.RAW_DATA_FILE.relative_to(config.PROJECT_ROOT)}`.",
                   title="Dataset missing", icon=":material/database_off:")
        st.code("python -m src.data.loader", language="bash")
        st.stop()
    return df


def require_results() -> dict[str, Any]:
    res = results()
    if res is None:
        st.info("No experiment results yet. Train the models to populate this page; nothing on it "
                "is shown until real results exist.", title="Models not trained",
                icon=":material/model_training:")
        st.code("python -m src.training.train          # full run (nested CV)\n"
                "python -m src.training.train --quick  # fast run", language="bash")
        st.stop()
    return res


def require_model() -> LoadedModel:
    m = model()
    if m is None:
        st.info("No verified model artifact is available. Train the models first; the dashboard "
                "only loads model files whose hash matches the training metadata.",
                title="Model unavailable", icon=":material/model_training:")
        st.code("python -m src.training.train", language="bash")
        st.stop()
    return m


def model_names(res: dict) -> dict[str, str]:
    return {k: m["display_name"] for k, m in res["models"].items()}
