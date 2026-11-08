"""Probability landscape: a model's chance of survival over two details at once.

For one passenger profile (sex, class, port and any detail not on an axis held
fixed) the model is asked for its chance of survival at every combination of two
numeric details, e.g. age x fare. The grid is the "landscape" drawn in 3D.

Like the what-if curves, this shows how *the model* behaves; it is not a causal
claim about what would have happened to a real person.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

import config
from src.prediction.predict import LoadedModel, predict_frame

# Plain label and the values scanned for each numeric detail. Ranges stay inside
# the data (fares above 200 are rare luxury tickets and would squash the view).
AXES: dict[str, tuple[str, np.ndarray]] = {
    "age": ("Age (years)", np.arange(0, 81, 2.0)),
    "fare": ("Fare paid (£, 1912)", np.arange(0, 201, 5.0)),
    "sibsp": ("Siblings / spouses aboard", np.arange(0, 9)),
    "parch": ("Parents / children aboard", np.arange(0, 7)),
}


def probability_grid(model: LoadedModel, base: dict[str, Any], x_feature: str,
                     y_feature: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Chance of survival at every (x, y) combination.

    ``base`` holds a value for every model input; the two axis features are
    overwritten. Returns ``(xs, ys, z)`` where ``z`` has shape ``(len(ys), len(xs))``.
    """
    for name in (x_feature, y_feature):
        if name not in AXES:
            raise KeyError(f"Unknown axis {name!r}; choose from {list(AXES)}")
    if x_feature == y_feature:
        raise ValueError("Choose two different details for the two axes.")
    xs, ys = AXES[x_feature][1], AXES[y_feature][1]
    xx, yy = np.meshgrid(xs, ys)
    frame = pd.DataFrame([base] * xx.size)[config.MODEL_FEATURES]
    frame[x_feature] = xx.ravel()
    frame[y_feature] = yy.ravel()
    prob = predict_frame(model, frame)["prob_survived"].to_numpy()
    return xs, ys, prob.reshape(xx.shape)


def matching_passengers(df: pd.DataFrame, base: dict[str, Any], x_feature: str,
                        y_feature: str) -> pd.DataFrame:
    """Real passengers with the same sex and ticket class and a known value on both axes."""
    same = df[(df["sex"] == base["sex"]) & (df["pclass"] == int(base["pclass"]))]
    same = same.dropna(subset=[x_feature, y_feature])
    return same[[x_feature, y_feature, config.TARGET]].rename(
        columns={x_feature: "x", y_feature: "y", config.TARGET: "survived"})
