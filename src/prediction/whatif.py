"""What-if analysis: how the prediction moves when one detail changes.

All other details are held at the passenger's current values (a "ceteris
paribus" curve). This shows the model's behaviour for one person; it is not a
causal effect and can differ from passenger to passenger.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

import config
from src.prediction.predict import LoadedModel, predict_frame

# Values swept for each detail. Numeric ranges stay inside the training data.
GRIDS: dict[str, list[Any]] = {
    "age": list(np.arange(0, 81, 1.0)),
    "fare": list(np.arange(0, 301, 2.5)),
    "sibsp": list(range(0, 9)),
    "parch": list(range(0, 7)),
    "pclass": [1, 2, 3],
    "sex": ["female", "male"],
    "embarked": ["C", "Q", "S"],
}
CATEGORICAL = {"pclass", "sex", "embarked"}


def what_if_curve(model: LoadedModel, passenger: dict[str, Any], feature: str) -> pd.DataFrame:
    """Return a frame with columns ``value`` and ``prob`` for each value of ``feature``."""
    if feature not in GRIDS:
        raise KeyError(f"Unknown feature {feature!r}; choose from {list(GRIDS)}")
    values = GRIDS[feature]
    rows = pd.DataFrame([{**passenger, feature: v} for v in values])
    out = predict_frame(model, rows[config.MODEL_FEATURES])
    return pd.DataFrame({"value": values, "prob": out["prob_survived"].to_numpy()})


def biggest_swing(model: LoadedModel, passenger: dict[str, Any]) -> list[tuple[str, float, float]]:
    """For each detail, the lowest and highest chance reachable by changing only that detail.

    Sorted by the size of the swing, largest first.
    """
    swings = []
    for feature in GRIDS:
        curve = what_if_curve(model, passenger, feature)
        swings.append((feature, float(curve["prob"].min()), float(curve["prob"].max())))
    return sorted(swings, key=lambda s: s[2] - s[1], reverse=True)
