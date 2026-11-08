"""Inference: safe model loading and individual / batch prediction.

Joblib files are pickles and can execute code when loaded, so models are only
loaded from the project's ``models/`` directory and only when their SHA-256
matches the digest recorded in ``models/metadata.json`` at training time.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

import config
from src.data.validation import DataValidationError, validate_features
from src.utils import file_sha256, get_logger, read_json

log = get_logger(__name__)


class ModelUnavailableError(RuntimeError):
    """Raised when no trained, verified model artifact is available."""


@dataclass
class LoadedModel:
    pipeline: Pipeline
    key: str
    display_name: str
    threshold: float
    metadata: dict[str, Any]


def load_metadata(models_dir: Path = config.MODELS_DIR) -> dict[str, Any]:
    path = models_dir / config.METADATA_FILE.name
    if not path.exists():
        raise ModelUnavailableError("No trained model found. Run `python -m src.training.train` first.")
    return read_json(path)


def load_model(filename: str = config.BEST_MODEL_FILE.name,
               models_dir: Path = config.MODELS_DIR) -> LoadedModel:
    """Load a fitted pipeline after verifying its location and hash."""
    models_dir = Path(models_dir).resolve()
    path = (models_dir / filename).resolve()
    if path.parent != models_dir or path.suffix != ".joblib":
        raise ModelUnavailableError(f"Refusing to load {filename!r}: not a model file in {models_dir}.")
    if not path.exists():
        raise ModelUnavailableError(f"Model file {path.name} not found. Train the models first.")
    meta = load_metadata(models_dir)
    expected = meta.get("model_sha256", {}).get(path.name)
    actual = file_sha256(path)
    if expected is None or expected != actual:
        raise ModelUnavailableError(
            f"Hash check failed for {path.name}; the file was not produced by the recorded "
            "training run. Re-train instead of loading an unverified pickle.")
    pipeline = joblib.load(path)
    if filename == config.BEST_MODEL_FILE.name:
        key, name = meta["selected_model"], meta["selected_display_name"]
    else:
        key = path.stem
        name = key.replace("_", " ").title()
    log.info("Model loaded", extra={"fields": {"file": path.name, "model": key}})
    return LoadedModel(pipeline, key, name, float(meta.get("decision_threshold", 0.5)), meta)


def predict_frame(model: LoadedModel, data: pd.DataFrame) -> pd.DataFrame:
    """Validate ``data`` and return it with probability and label columns.

    Raises :class:`DataValidationError` if any row is invalid; no partial
    predictions are produced, so a batch either succeeds entirely or reports
    every problem.
    """
    features = validate_features(data)
    prob = model.pipeline.predict_proba(features)[:, 1]
    out = data.copy()
    out["prob_survived"] = np.round(prob, 4)
    out["prob_not_survived"] = np.round(1 - prob, 4)
    out["prediction"] = np.where(prob >= model.threshold, config.POSITIVE_LABEL, config.NEGATIVE_LABEL)
    out["model"] = model.display_name
    return out


def predict_one(model: LoadedModel, passenger: dict[str, Any]) -> dict[str, Any]:
    """Predict for a single passenger given as a dict of raw feature values."""
    row = predict_frame(model, pd.DataFrame([passenger])).iloc[0]
    return {
        "prediction": row["prediction"],
        "prob_survived": float(row["prob_survived"]),
        "prob_not_survived": float(row["prob_not_survived"]),
        "model": model.display_name,
        "threshold": model.threshold,
    }


def read_batch_csv(source) -> pd.DataFrame:
    """Read an uploaded CSV, failing with a clear message on parse errors."""
    try:
        df = pd.read_csv(source)
    except Exception as exc:  # pandas raises several parser error types
        raise DataValidationError([f"Could not parse CSV: {exc}"]) from exc
    if len(df) > 100_000:
        raise DataValidationError(["Batch is limited to 100,000 rows."])
    return df


__all__ = ["DataValidationError", "LoadedModel", "ModelUnavailableError", "load_metadata",
           "load_model", "predict_frame", "predict_one", "read_batch_csv"]
