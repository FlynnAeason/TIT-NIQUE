"""Centralised project configuration.

Every path is resolved relative to this file, so the project never depends on
the current working directory. Values that a user may reasonably want to change
(seed, split size, search budget) can be overridden through environment
variables or the training CLI.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT: Path = Path(__file__).resolve().parent
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
MODELS_DIR: Path = PROJECT_ROOT / "models"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"

RAW_DATA_FILE: Path = RAW_DATA_DIR / "titanic.csv"
RESULTS_FILE: Path = REPORTS_DIR / "experiment_results.json"
TEST_PREDICTIONS_FILE: Path = REPORTS_DIR / "test_predictions.csv"
METADATA_FILE: Path = MODELS_DIR / "metadata.json"
BEST_MODEL_FILE: Path = MODELS_DIR / "best_model.joblib"
PDF_REPORT_FILE: Path = REPORTS_DIR / "titanic_model_report.pdf"

# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
# The notebook loads ``sns.load_dataset('titanic')``; seaborn fetches this exact
# file from the seaborn-data repository. We download it once and pin its hash so
# results are reproducible and the file cannot silently change.
DATA_SOURCE_URL: str = (
    "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv"
)
DATA_SHA256: str = "81787d320d7f7b03df935e91de8bd19e11d45c5bbcab86ef4d4a76dc91b7d4f2"

TARGET: str = "survived"
POSITIVE_LABEL: str = "Survived"
NEGATIVE_LABEL: str = "Did not survive"

# Raw columns expected in the source file.
EXPECTED_COLUMNS: tuple[str, ...] = (
    "survived", "pclass", "sex", "age", "sibsp", "parch", "fare", "embarked",
    "class", "who", "adult_male", "deck", "embark_town", "alive", "alone",
)

# Raw predictors fed to the model pipeline (engineered features are derived
# inside the pipeline from these).
NUMERIC_FEATURES: list[str] = ["age", "sibsp", "parch", "fare"]
CATEGORICAL_FEATURES: list[str] = ["pclass", "sex", "embarked"]
MODEL_FEATURES: list[str] = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Engineered row-wise features (no learned statistics -> leakage-safe).
CHILD_AGE_THRESHOLD: float = 16.0  # matches seaborn's derivation of `who`

# Columns excluded from modelling, each with an explicit reason. The redundancy
# claims are re-verified from the data at runtime (see src.data.profiling).
EXCLUDED_FEATURES: dict[str, str] = {
    "alive": "Target leakage: 'yes'/'no' restatement of `survived`.",
    "class": "Exact duplicate of `pclass` (First/Second/Third <-> 1/2/3).",
    "who": "Deterministic function of `sex` and `age < 16`; replaced by the "
           "engineered `is_child` flag plus a missing-age indicator.",
    "adult_male": "Deterministic function of `who` (adult_male == who=='man').",
    "alone": "Deterministic function of `sibsp + parch == 0`; captured by the "
             "engineered `family_size`.",
    "embark_town": "Exact duplicate of `embarked` (C/Q/S <-> town names).",
    "deck": "~77% missing; imputing it would mostly encode missingness, which "
            "is itself strongly tied to passenger class.",
}

# Valid values for categorical inputs (used for validation and the UI).
CATEGORY_LEVELS: dict[str, list] = {
    "pclass": [1, 2, 3],
    "sex": ["female", "male"],
    "embarked": ["C", "Q", "S"],
}
EMBARKED_NAMES: dict[str, str] = {"C": "Cherbourg", "Q": "Queenstown", "S": "Southampton"}

# Plausible numeric ranges (validation only; slightly wider than observed data).
NUMERIC_RANGES: dict[str, tuple[float, float]] = {
    "age": (0.0, 100.0),
    "sibsp": (0, 10),
    "parch": (0, 10),
    "fare": (0.0, 1000.0),
}


# --------------------------------------------------------------------------- #
# Experiment
# --------------------------------------------------------------------------- #
def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return int(value) if value not in (None, "") else default


@dataclass(frozen=True)
class ExperimentConfig:
    """Settings that control splitting, tuning and evaluation."""

    random_state: int = field(default_factory=lambda: _env_int("TITANIC_SEED", 42))
    test_size: float = 0.20
    cv_folds: int = 5
    # Search budget per model (RandomizedSearchCV iterations; grids smaller than
    # this are searched exhaustively).
    n_iter: int = field(default_factory=lambda: _env_int("TITANIC_N_ITER", 25))
    # Nested CV gives an (approximately) unbiased estimate of the *whole* tuning
    # procedure and is used for model selection. Disable for a quick run.
    nested_cv: bool = True
    outer_folds: int = 5
    # Primary selection metric, fixed *before* looking at test results.
    selection_metric: str = "roc_auc"
    n_bootstrap: int = 2000
    permutation_repeats: int = 30
    n_jobs: int = -1
    decision_threshold: float = 0.5


SELECTION_METRIC_RATIONALE: str = (
    "ROC-AUC measures how well a model ranks survivors above non-survivors "
    "across every possible threshold. It is insensitive to the moderate class "
    "imbalance (~38% survived) and does not depend on the arbitrary 0.5 cut-off, "
    "so it compares the models' underlying discrimination rather than one "
    "operating point. Ties are broken by lower cross-validation variance."
)

# Okabe-Ito colour-blind-safe palette, shared by Plotly and Matplotlib charts.
PALETTE: dict[str, str] = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "vermillion": "#D55E00",
    "sky": "#56B4E9",
    "purple": "#CC79A7",
    "yellow": "#F0E442",
    "grey": "#6B7280",
}
CLASS_COLORS: dict[int, str] = {0: PALETTE["vermillion"], 1: PALETTE["blue"]}
MODEL_COLORS: list[str] = [PALETTE["blue"], PALETTE["orange"], PALETTE["green"],
                           PALETTE["purple"], PALETTE["sky"], PALETTE["vermillion"]]


def ensure_directories() -> None:
    """Create all output directories if they do not exist."""
    for path in (RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, REPORTS_DIR):
        path.mkdir(parents=True, exist_ok=True)
