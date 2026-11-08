"""Baseline ladder: how much does the model add over simple rules?

Each rung is a rule a person could apply by hand. Rules that need a decision
(which groups "survive") are learned from the **training** passengers only, by
majority vote within each group, and then scored on the same untouched test
passengers as the model.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

import config
from src.evaluation.metrics import compute_metrics


def _group_rule(train: pd.DataFrame, test: pd.DataFrame, keys: list[str]) -> tuple[np.ndarray, str]:
    """Predict the majority outcome of each group (learned on train); unseen groups -> 'died'."""
    rates = train.groupby(keys, observed=True)[config.TARGET].mean()
    survive_groups = rates[rates > 0.5].index
    idx = pd.MultiIndex.from_frame(test[keys]) if len(keys) > 1 else pd.Index(test[keys[0]])
    pred = idx.isin(survive_groups).astype(int)
    if len(keys) == 1:
        names = [str(g) for g in survive_groups]
    else:
        names = [" & ".join(str(v) for v in g) for g in survive_groups]
    return pred, ", ".join(names) if names else "nobody"


def baseline_ladder(df: pd.DataFrame, test_index: pd.Index, model_name: str,
                    model_metrics: dict[str, float]) -> list[dict[str, Any]]:
    """Accuracy (and recall/specificity) for each rung, ending with the trained model.

    ``df`` is the raw validated dataset; ``test_index`` are the row labels of the
    test passengers (from ``data/processed/split_indices.csv``).
    """
    data = df.copy()
    data["is_child"] = np.where(data["age"] < config.CHILD_AGE_THRESHOLD, "child", "adult")
    data["class"] = "class " + data["pclass"].astype(str)
    is_test = data.index.isin(test_index)
    train, test = data[~is_test], data[is_test]
    y = test[config.TARGET].to_numpy()

    rungs: list[tuple[str, str, np.ndarray]] = [
        ("Guess 'did not survive' for everyone", "Nobody survives.", np.zeros(len(test), dtype=int)),
    ]
    for label, keys in (("Women survive, men don't", ["sex"]),
                        ("Women and children survive", ["sex", "is_child"]),
                        ("Use sex and ticket class", ["sex", "class"])):
        pred, groups = _group_rule(train, test, keys)
        rungs.append((label, f"Groups predicted to survive (majority in training data): {groups}.", pred))

    out = []
    for label, detail, pred in rungs:
        m = compute_metrics(y, pred.astype(float))
        out.append({"step": label, "detail": detail, "accuracy": m["accuracy"], "recall": m["recall"],
                    "specificity": m["specificity"], "correct": int((pred == y).sum()), "n": len(y),
                    "is_model": False})
    out.append({"step": f"Our model ({model_name})", "detail": "All seven details, learned automatically.",
                "accuracy": model_metrics["accuracy"], "recall": model_metrics["recall"],
                "specificity": model_metrics["specificity"],
                "correct": int(round(model_metrics["accuracy"] * len(y))), "n": len(y), "is_model": True})
    return out
