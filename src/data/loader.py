"""Loading the Titanic dataset from a pinned, hash-verified local copy."""

from __future__ import annotations

import urllib.request
from pathlib import Path

import pandas as pd

import config
from src.data.validation import validate_raw_dataset
from src.utils import file_sha256, get_logger

log = get_logger(__name__)


class DatasetNotFoundError(FileNotFoundError):
    """Raised when the raw dataset is not available locally."""


def download_dataset(path: Path = config.RAW_DATA_FILE, url: str = config.DATA_SOURCE_URL,
                     expected_sha256: str | None = config.DATA_SHA256) -> Path:
    """Download the seaborn Titanic CSV and verify its SHA-256 digest.

    The file is the one ``seaborn.load_dataset('titanic')`` uses in the original
    notebook. A mismatching hash deletes the download and raises ``ValueError``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    log.info("Downloading dataset", extra={"fields": {"url": url}})
    with urllib.request.urlopen(url, timeout=30) as response:  # noqa: S310 - fixed https URL
        tmp.write_bytes(response.read())
    digest = file_sha256(tmp)
    if expected_sha256 and digest != expected_sha256.lower():
        tmp.unlink(missing_ok=True)
        raise ValueError(f"Downloaded file hash {digest} does not match pinned {expected_sha256}.")
    tmp.replace(path)
    log.info("Dataset saved", extra={"fields": {"path": path, "sha256": digest[:12]}})
    return path


def load_raw_data(path: Path = config.RAW_DATA_FILE, validate: bool = True,
                  verify_hash: bool = False) -> pd.DataFrame:
    """Load the raw Titanic CSV.

    Args:
        path: CSV location (defaults to ``data/raw/titanic.csv``).
        validate: run schema validation and dtype normalisation.
        verify_hash: additionally require the file to match the pinned hash.

    Raises:
        DatasetNotFoundError: if the file does not exist.
        DataValidationError: if the schema is invalid.
    """
    path = Path(path)
    if not path.exists():
        raise DatasetNotFoundError(
            f"Dataset not found at {path}. Run `python -m src.data.loader` to download it "
            f"from {config.DATA_SOURCE_URL}."
        )
    if verify_hash:
        digest = file_sha256(path)
        if digest != config.DATA_SHA256.lower():
            raise ValueError(f"Dataset hash {digest} differs from the pinned version.")
    df = pd.read_csv(path)
    log.info("Loaded raw data", extra={"fields": {"rows": len(df), "cols": df.shape[1]}})
    return validate_raw_dataset(df) if validate else df


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return the model feature matrix and target, excluding leaking columns."""
    return df[config.MODEL_FEATURES].copy(), df[config.TARGET].astype(int).copy()


if __name__ == "__main__":
    download_dataset()
