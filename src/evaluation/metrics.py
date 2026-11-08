"""Classification metrics, curves, bootstrap intervals and threshold analysis."""

from __future__ import annotations

import warnings
from typing import Any, Callable

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    make_scorer,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

METRIC_LABELS: dict[str, str] = {
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall (sensitivity)",
    "specificity": "Specificity",
    "f1": "F1-score",
    "balanced_accuracy": "Balanced accuracy",
    "roc_auc": "ROC-AUC",
    "average_precision": "PR-AUC (average precision)",
    "mcc": "Matthews corr. coef.",
    "brier": "Brier score",
    "log_loss": "Log loss",
}

METRIC_EXPLANATIONS: dict[str, str] = {
    "accuracy": "Share of all passengers classified correctly. Easy to read but can "
                "flatter a model on imbalanced data (predicting 'did not survive' for "
                "everyone already scores ~62%).",
    "precision": "Of passengers predicted to survive, the share who actually survived.",
    "recall": "Of passengers who survived, the share the model identified.",
    "specificity": "Of passengers who died, the share correctly predicted as not surviving.",
    "f1": "Harmonic mean of precision and recall for the 'survived' class.",
    "balanced_accuracy": "Mean of recall and specificity; 0.5 = chance regardless of imbalance.",
    "roc_auc": "Probability that a random survivor is ranked above a random non-survivor. "
               "Threshold-independent; 0.5 = chance. Primary selection metric.",
    "average_precision": "Area under the precision-recall curve; its chance level equals "
                         "the survival rate (~0.38), not 0.5.",
    "mcc": "Correlation between predictions and truth using all four confusion-matrix "
           "cells; ranges -1 to 1, 0 = chance. Robust to imbalance.",
    "brier": "Mean squared error of predicted probabilities (lower is better); "
             "measures calibration and sharpness together.",
    "log_loss": "Penalises confident wrong probabilities (lower is better).",
}

HIGHER_IS_BETTER: dict[str, bool] = {k: k not in {"brier", "log_loss"} for k in METRIC_LABELS}


def _specificity(y_true, y_pred) -> float:
    tn, fp, _, _ = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return float(tn / (tn + fp)) if (tn + fp) else float("nan")


def compute_metrics(y_true, y_prob, threshold: float = 0.5) -> dict[str, float]:
    """Compute all classification metrics from true labels and P(survived).

    Metrics that are undefined for the given data (e.g. ROC-AUC with a single
    class present, precision with no positive predictions) are returned as
    ``nan`` rather than silently replaced by 0.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)
    both_classes = len(np.unique(y_true)) == 2
    has_pos_pred = y_pred.sum() > 0
    has_pos_true = y_true.sum() > 0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=np.nan) if has_pos_pred else np.nan,
            "recall": recall_score(y_true, y_pred, zero_division=np.nan) if has_pos_true else np.nan,
            "specificity": _specificity(y_true, y_pred),
            "f1": f1_score(y_true, y_pred, zero_division=np.nan) if (has_pos_pred or has_pos_true) else np.nan,
            "balanced_accuracy": balanced_accuracy_score(y_true, y_pred) if both_classes else np.nan,
            "roc_auc": roc_auc_score(y_true, y_prob) if both_classes else np.nan,
            "average_precision": average_precision_score(y_true, y_prob) if has_pos_true else np.nan,
            "mcc": matthews_corrcoef(y_true, y_pred) if both_classes else np.nan,
            "brier": brier_score_loss(y_true, y_prob),
            "log_loss": log_loss(y_true, np.clip(y_prob, 1e-15, 1 - 1e-15), labels=[0, 1]),
        }
    return {k: float(v) for k, v in out.items()}


def cv_scorers() -> dict[str, Any]:
    """Scorers for ``cross_validate`` matching :func:`compute_metrics` names."""
    return {
        "roc_auc": "roc_auc",
        "average_precision": "average_precision",
        "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy",
        "f1": "f1",
        "precision": make_scorer(precision_score, zero_division=0),
        "recall": "recall",
        "specificity": make_scorer(_specificity),
        "mcc": make_scorer(matthews_corrcoef),
        "brier": "neg_brier_score",
    }


def confusion(y_true, y_pred) -> dict[str, int]:
    """Confusion-matrix cells as a dict (labels 0 = died, 1 = survived)."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def report_dict(y_true, y_pred) -> dict[str, Any]:
    """sklearn classification report as a dict with readable class names."""
    return classification_report(y_true, y_pred, labels=[0, 1],
                                 target_names=["Did not survive", "Survived"],
                                 output_dict=True, zero_division=0)


def curves(y_true, y_prob, max_points: int = 200) -> dict[str, Any]:
    """ROC and precision-recall curve coordinates (down-sampled for storage)."""
    fpr, tpr, roc_thr = roc_curve(y_true, y_prob)
    prec, rec, pr_thr = precision_recall_curve(y_true, y_prob)

    def thin(*arrays):
        n = len(arrays[0])
        if n <= max_points:
            return arrays
        idx = np.unique(np.linspace(0, n - 1, max_points).round().astype(int))
        return tuple(a[idx] for a in arrays)

    fpr, tpr, roc_thr = thin(fpr, tpr, np.r_[roc_thr])
    prec, rec = thin(prec, rec)
    return {"roc": {"fpr": fpr, "tpr": tpr, "threshold": np.clip(roc_thr, 0, 1)},
            "pr": {"precision": prec, "recall": rec},
            "prevalence": float(np.mean(y_true))}


def _safe_div(a: float, b: float) -> float:
    return a / b if b else float("nan")


def fast_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5,
                 names: tuple[str, ...] | None = None) -> dict[str, float]:
    """Numpy implementation of :func:`compute_metrics` for bootstrap loops.

    Produces the same values (verified in tests) roughly 10x faster by computing
    the confusion matrix directly and skipping sklearn's input validation.
    ``names`` restricts the (costlier) ranking metrics computed.
    """
    pred = y_prob >= threshold
    pos = y_true == 1
    tp = float(np.sum(pred & pos))
    fp = float(np.sum(pred & ~pos))
    fn = float(np.sum(~pred & pos))
    tn = float(np.sum(~pred & ~pos))
    n = tp + fp + fn + tn
    both = (tp + fn) > 0 and (tn + fp) > 0
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    spec = _safe_div(tn, tn + fp)
    denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    out = {
        "accuracy": (tp + tn) / n,
        "precision": precision,
        "recall": recall,
        "specificity": spec,
        "f1": _safe_div(2 * tp, 2 * tp + fp + fn) if (tp + fp + fn) else float("nan"),
        "balanced_accuracy": (recall + spec) / 2 if both else float("nan"),
        "mcc": ((tp * tn - fp * fn) / denom if denom else 0.0) if both else float("nan"),
        "brier": float(np.mean((y_prob - y_true) ** 2)),
    }
    want = names or ("roc_auc", "average_precision")
    if "roc_auc" in want:
        out["roc_auc"] = roc_auc_score(y_true, y_prob) if both else float("nan")
    if "average_precision" in want:
        out["average_precision"] = average_precision_score(y_true, y_prob) if tp + fn else float("nan")
    return out


def bootstrap_ci(y_true, y_prob, metric_fns: dict[str, Callable] | None = None,
                 n_boot: int = 2000, alpha: float = 0.05, seed: int = 42,
                 threshold: float = 0.5) -> dict[str, dict[str, float]]:
    """Percentile bootstrap confidence intervals for test-set metrics.

    Resamples test rows with replacement (stratified by class so every replicate
    keeps the original class balance and threshold metrics stay defined). This
    quantifies uncertainty due to the finite *test sample* only; it does not
    capture variability from re-training on different data.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y_true == 1), np.flatnonzero(y_true == 0)
    names = list(metric_fns) if metric_fns else [
        "accuracy", "precision", "recall", "specificity", "f1", "balanced_accuracy",
        "roc_auc", "average_precision", "mcc", "brier"]
    samples: dict[str, list[float]] = {k: [] for k in names}
    for _ in range(n_boot):
        idx = np.r_[rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]
        m = fast_metrics(y_true[idx], y_prob[idx], threshold, tuple(names))
        for k in names:
            samples[k].append(m[k])
    point = compute_metrics(y_true, y_prob, threshold)
    out = {}
    for k, vals in samples.items():
        arr = np.asarray(vals, dtype=float)
        arr = arr[np.isfinite(arr)]
        lo, hi = (np.quantile(arr, [alpha / 2, 1 - alpha / 2]) if len(arr) else (np.nan, np.nan))
        out[k] = {"estimate": point[k], "low": float(lo), "high": float(hi),
                  "se": float(arr.std(ddof=1)) if len(arr) > 1 else float("nan")}
    return out


def paired_bootstrap_difference(y_true, prob_a, prob_b, metric: str = "roc_auc",
                                n_boot: int = 2000, seed: int = 42) -> dict[str, float]:
    """Bootstrap CI for metric(A) - metric(B) on the same resampled test rows.

    Pairing removes the shared test-sample noise, so this answers "is model A
    distinguishable from model B on this test set?" more sharply than comparing
    two independent intervals.
    """
    y_true = np.asarray(y_true).astype(int)
    prob_a, prob_b = np.asarray(prob_a, float), np.asarray(prob_b, float)
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y_true == 1), np.flatnonzero(y_true == 0)
    diffs = []
    for _ in range(n_boot):
        idx = np.r_[rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]
        a = fast_metrics(y_true[idx], prob_a[idx], names=(metric,))[metric]
        b = fast_metrics(y_true[idx], prob_b[idx], names=(metric,))[metric]
        diffs.append(a - b)
    diffs = np.asarray(diffs)
    point = compute_metrics(y_true, prob_a)[metric] - compute_metrics(y_true, prob_b)[metric]
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    return {"difference": float(point), "low": float(lo), "high": float(hi),
            "share_a_better": float(np.mean(diffs > 0))}


def threshold_table(y_true, y_prob, thresholds=None) -> list[dict[str, float]]:
    """Metrics across decision thresholds (for out-of-fold training predictions)."""
    thresholds = np.round(np.arange(0.05, 0.96, 0.05), 2) if thresholds is None else thresholds
    rows = []
    for t in thresholds:
        m = compute_metrics(y_true, y_prob, threshold=float(t))
        m["threshold"] = float(t)
        m["predicted_positive_rate"] = float(np.mean(np.asarray(y_prob) >= t))
        rows.append(m)
    return rows
