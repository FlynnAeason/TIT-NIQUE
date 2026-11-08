"""Schema and value validation shared by training and inference."""

from __future__ import annotations

import numpy as np
import pandas as pd

import config


class DataValidationError(ValueError):
    """Raised when input data does not satisfy the expected schema.

    ``errors`` holds one human-readable message per problem so the UI can show
    all of them at once instead of failing on the first.
    """

    def __init__(self, errors: list[str]):
        self.errors = list(errors)
        super().__init__("; ".join(self.errors))


# Features that must be present for a meaningful prediction. Age, fare and port
# may be missing: the pipeline imputes them exactly as it did during training.
REQUIRED_NON_NULL: tuple[str, ...] = ("pclass", "sex", "sibsp", "parch")


def validate_raw_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Validate the raw training file and return it with normalised dtypes.

    Checks: expected columns, non-empty, binary non-null target, and that model
    features satisfy :func:`validate_features`.
    """
    errors: list[str] = []
    if df.empty:
        raise DataValidationError(["Dataset is empty."])

    missing = [c for c in config.EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        errors.append(f"Missing expected columns: {missing}")
    if config.TARGET in df.columns:
        target = df[config.TARGET]
        if target.isna().any():
            errors.append(f"Target '{config.TARGET}' contains missing values.")
        elif not set(pd.unique(target)).issubset({0, 1}):
            errors.append(f"Target '{config.TARGET}' must be binary 0/1.")
    if errors:
        raise DataValidationError(errors)

    out = df.copy()
    out[config.TARGET] = out[config.TARGET].astype(int)
    out[config.MODEL_FEATURES] = validate_features(out[config.MODEL_FEATURES])
    return out


def validate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and coerce model input features.

    Returns a new DataFrame containing exactly ``config.MODEL_FEATURES`` with
    canonical dtypes (numeric columns as float, ``pclass`` as int, text columns
    as stripped strings; missing values as ``NaN``). Raises
    :class:`DataValidationError` listing every problem found.
    """
    errors: list[str] = []
    if not isinstance(df, pd.DataFrame):
        raise DataValidationError(["Input must be a pandas DataFrame."])
    if df.empty:
        raise DataValidationError(["Input contains no rows."])

    absent = [c for c in config.MODEL_FEATURES if c not in df.columns]
    if absent:
        raise DataValidationError([
            f"Missing required columns: {absent}. Expected: {config.MODEL_FEATURES}"
        ])

    out = pd.DataFrame(index=df.index)

    # Numeric columns -------------------------------------------------------
    for col in config.NUMERIC_FEATURES:
        raw = df[col].replace({"": np.nan})
        numeric = pd.to_numeric(raw, errors="coerce")
        bad = numeric.isna() & raw.notna()
        if bad.any():
            errors.append(f"'{col}' has non-numeric values in rows {_rows(bad)}.")
        lo, hi = config.NUMERIC_RANGES[col]
        out_of_range = numeric.notna() & ((numeric < lo) | (numeric > hi))
        if out_of_range.any():
            errors.append(f"'{col}' must be within [{lo}, {hi}] (rows {_rows(out_of_range)}).")
        if col in ("sibsp", "parch"):
            non_int = numeric.notna() & (numeric % 1 != 0)
            if non_int.any():
                errors.append(f"'{col}' must be a whole number (rows {_rows(non_int)}).")
        out[col] = numeric.astype(float)

    # pclass ----------------------------------------------------------------
    pclass = pd.to_numeric(df["pclass"].replace({"": np.nan}), errors="coerce")
    bad = pclass.notna() & ~pclass.isin(config.CATEGORY_LEVELS["pclass"])
    bad |= pclass.isna() & df["pclass"].notna()
    if bad.any():
        errors.append(f"'pclass' must be one of {config.CATEGORY_LEVELS['pclass']} (rows {_rows(bad)}).")
    out["pclass"] = pclass

    # Text categoricals -----------------------------------------------------
    for col, transform in (("sex", str.lower), ("embarked", str.upper)):
        series = df[col].astype("object").where(df[col].notna(), None)
        series = series.map(lambda v: transform(str(v).strip()) if v is not None else None)
        series = series.replace({"": None, "NAN": None, "nan": None})
        allowed = config.CATEGORY_LEVELS[col]
        bad = series.notna() & ~series.isin(allowed)
        if bad.any():
            errors.append(f"'{col}' must be one of {allowed} (rows {_rows(bad)}).")
        out[col] = series.astype("object").where(series.notna(), np.nan)

    for col in REQUIRED_NON_NULL:
        if col in out and out[col].isna().any():
            errors.append(f"'{col}' is required but missing in rows {_rows(out[col].isna())}.")

    if errors:
        raise DataValidationError(errors)

    out["pclass"] = out["pclass"].astype(int)
    for col in config.NUMERIC_FEATURES:
        out[col] = out[col].astype(float)
    return out[config.MODEL_FEATURES]


def _rows(mask: pd.Series, limit: int = 5) -> str:
    """Format the (1-based positional) rows where ``mask`` is True."""
    positions = np.flatnonzero(mask.to_numpy()) + 1
    shown = ", ".join(str(p) for p in positions[:limit])
    return shown + (f", ... ({len(positions)} total)" if len(positions) > limit else "")
