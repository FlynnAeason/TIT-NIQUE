"""End-to-end training workflow.

Run from the project root:

    python -m src.training.train            # full run (nested CV)
    python -m src.training.train --quick    # fast smoke run

Workflow
--------
1. Load and validate the raw data; profile it.
2. Stratified train/test split. The test set is not touched again until every
   model has been tuned *and* the best model has been selected.
3. For each candidate: RandomizedSearchCV (stratified K-fold, training data
   only) with ROC-AUC as the refit metric. When ``nested_cv`` is on, the whole
   search is wrapped in an outer stratified CV to obtain an approximately
   unbiased estimate of the tuned model's performance; this is the score used
   for model selection.
4. Select the model with the best selection score (ties -> lower std).
5. Only then evaluate every tuned model on the untouched test set, with
   bootstrap CIs, curves, confusion matrices and interpretation.
6. Persist fitted pipelines, metadata (versions, hashes) and results.
"""

from __future__ import annotations

import argparse
import platform
import shutil
import sys
import time
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
    learning_curve,
    train_test_split,
)

import config
from src.data.loader import load_raw_data, split_features_target
from src.data.profiling import full_profile
from src.evaluation.interpretation import model_importance, permutation_importance_table
from src.evaluation.metrics import (
    bootstrap_ci,
    compute_metrics,
    confusion,
    curves,
    cv_scorers,
    paired_bootstrap_difference,
    report_dict,
    threshold_table,
)
from src.features.engineering import FEATURE_DESCRIPTIONS
from src.preprocessing.pipeline import build_pipeline
from src.training.models import ModelSpec, get_model_specs
from src.utils import file_sha256, get_logger, write_json

log = get_logger("training.train")  # stable name even when run with `python -m`

CV_METRICS = list(cv_scorers())


def split_data(X: pd.DataFrame, y: pd.Series, cfg: config.ExperimentConfig):
    """Stratified hold-out split (seeded)."""
    return train_test_split(X, y, test_size=cfg.test_size, stratify=y,
                            random_state=cfg.random_state)


def _search_space_size(dists: dict[str, Any]) -> float:
    size = 1.0
    for v in dists.values():
        if isinstance(v, list):
            size *= len(v)
        else:
            return float("inf")
    return size


def _make_search(spec: ModelSpec, cfg: config.ExperimentConfig) -> RandomizedSearchCV:
    inner = StratifiedKFold(cfg.cv_folds, shuffle=True, random_state=cfg.random_state)
    n_iter = int(min(cfg.n_iter, _search_space_size(spec.param_distributions)))
    return RandomizedSearchCV(
        build_pipeline(spec.factory(cfg.random_state)),
        spec.param_distributions,
        n_iter=n_iter,
        scoring=cv_scorers(),
        refit=cfg.selection_metric,
        cv=inner,
        random_state=cfg.random_state,
        n_jobs=cfg.n_jobs,
        error_score="raise",
    )


def _fix_sign(metric: str, values):
    """sklearn's Brier scorer is negated; flip it back to the usual sign."""
    arr = np.asarray(values, dtype=float)
    return -arr if metric == "brier" else arr


def tune_model(spec: ModelSpec, X_train: pd.DataFrame, y_train: pd.Series,
               cfg: config.ExperimentConfig) -> dict[str, Any]:
    """Tune one model on training data only; return the fitted search and CV stats."""
    t0 = time.perf_counter()
    nested: dict[str, Any] | None = None
    if cfg.nested_cv:
        outer = StratifiedKFold(cfg.outer_folds, shuffle=True, random_state=cfg.random_state + 1)
        cv_out = cross_validate(_make_search(spec, cfg), X_train, y_train, cv=outer,
                                scoring=cv_scorers(), n_jobs=1)
        nested = {m: _fix_sign(m, cv_out[f"test_{m}"]).tolist() for m in CV_METRICS}
        log.info("Nested CV done", extra={"fields": {
            "model": spec.key, "roc_auc": f"{np.mean(nested['roc_auc']):.4f}"}})

    search = _make_search(spec, cfg)
    search.fit(X_train, y_train)
    res = search.cv_results_
    best = search.best_index_
    n_splits = search.n_splits_
    best_folds = {m: _fix_sign(m, [res[f"split{i}_test_{m}"][best] for i in range(n_splits)]).tolist()
                  for m in CV_METRICS}

    table = pd.DataFrame(res["params"])
    table.columns = [c.replace("classifier__", "").replace("estimator__", "") for c in table.columns]
    table = table.astype(object).where(table.notna(), "None").astype(str)
    for m in ("roc_auc", "accuracy", "f1", "balanced_accuracy"):
        table[f"mean_{m}"] = _fix_sign(m, res[f"mean_test_{m}"])
        table[f"std_{m}"] = res[f"std_test_{m}"]
    table["rank"] = res[f"rank_test_{cfg.selection_metric}"]
    table["fit_time_s"] = res["mean_fit_time"]
    table = table.sort_values("rank").reset_index(drop=True)

    sel_scores = nested[cfg.selection_metric] if nested else best_folds[cfg.selection_metric]
    return {
        "search": search,
        "best_params": {k.replace("classifier__", "").replace("estimator__", ""): v for k, v in search.best_params_.items()},
        "inner_cv_best": best_folds,
        "nested_cv": nested,
        "search_table": table,
        "selection_scores": sel_scores,
        "selection_mean": float(np.mean(sel_scores)),
        "selection_std": float(np.std(sel_scores, ddof=1)),
        "n_candidates": int(len(res["params"])),
        "tuning_seconds": time.perf_counter() - t0,
    }


def select_best(summaries: dict[str, dict[str, Any]]) -> str:
    """Pick the model with the highest selection score; ties -> lower std."""
    return max(summaries, key=lambda k: (round(summaries[k]["selection_mean"], 4),
                                         -summaries[k]["selection_std"]))


def _environment() -> dict[str, str]:
    import joblib as jl
    import scipy
    return {"python": sys.version.split()[0], "platform": platform.platform(),
            "scikit-learn": sklearn.__version__, "pandas": pd.__version__,
            "numpy": np.__version__, "scipy": scipy.__version__, "joblib": jl.__version__}


def run_training(cfg: config.ExperimentConfig | None = None,
                 model_keys: list[str] | None = None,
                 data_path: Path = config.RAW_DATA_FILE,
                 models_dir: Path = config.MODELS_DIR,
                 reports_dir: Path = config.REPORTS_DIR,
                 processed_dir: Path = config.PROCESSED_DATA_DIR) -> dict[str, Any]:
    """Execute the complete experiment and write all artifacts.

    Returns the results dictionary that is also written to
    ``reports/experiment_results.json``.
    """
    cfg = cfg or config.ExperimentConfig()
    started = time.perf_counter()
    for d in (models_dir, reports_dir, processed_dir):
        d.mkdir(parents=True, exist_ok=True)
    log.info("Experiment start", extra={"fields": asdict(cfg)})

    df = load_raw_data(data_path)
    profile = full_profile(df)
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = split_data(X, y, cfg)
    pd.DataFrame({"row_index": X.index, "split": np.where(X.index.isin(X_test.index), "test", "train")}
                 ).to_csv(processed_dir / "split_indices.csv", index=False)
    log.info("Split", extra={"fields": {"train": len(X_train), "test": len(X_test),
                                        "train_pos_rate": f"{y_train.mean():.3f}",
                                        "test_pos_rate": f"{y_test.mean():.3f}"}})

    # ---- Phase 1: tuning & selection on training data only ----------------
    specs = get_model_specs(model_keys)
    tuned: dict[str, dict[str, Any]] = {}
    for spec in specs:
        log.info("Tuning", extra={"fields": {"model": spec.key}})
        tuned[spec.key] = tune_model(spec, X_train, y_train, cfg)
    best_key = select_best(tuned)
    log.info("Selected (before test evaluation)", extra={"fields": {
        "model": best_key, cfg.selection_metric: f"{tuned[best_key]['selection_mean']:.4f}"}})

    # ---- Phase 2: evaluation on the untouched test set --------------------
    inner = StratifiedKFold(cfg.cv_folds, shuffle=True, random_state=cfg.random_state)
    model_results: dict[str, Any] = {}
    test_probs: dict[str, np.ndarray] = {}
    model_hashes: dict[str, str] = {}
    for spec in specs:
        t = tuned[spec.key]
        est = t["search"].best_estimator_
        prob = est.predict_proba(X_test)[:, 1]
        pred = (prob >= cfg.decision_threshold).astype(int)
        test_probs[spec.key] = prob
        path = models_dir / f"{spec.key}.joblib"
        joblib.dump(est, path)
        model_hashes[path.name] = file_sha256(path)
        model_results[spec.key] = {
            "key": spec.key,
            "display_name": spec.display_name,
            "description": spec.description,
            "tags": list(spec.tags),
            "search_space": {k.replace("classifier__", "").replace("estimator__", ""): (v if isinstance(v, list) else repr(v))
                             for k, v in spec.param_distributions.items()},
            "best_params": t["best_params"],
            "n_candidates": t["n_candidates"],
            "tuning_seconds": t["tuning_seconds"],
            "inner_cv_best": t["inner_cv_best"],
            "nested_cv": t["nested_cv"],
            "selection_scores": t["selection_scores"],
            "selection_mean": t["selection_mean"],
            "selection_std": t["selection_std"],
            "search_table": t["search_table"].head(25).to_dict(orient="list"),
            "test_metrics": compute_metrics(y_test, prob, cfg.decision_threshold),
            "test_ci": bootstrap_ci(y_test, prob, n_boot=cfg.n_bootstrap, seed=cfg.random_state,
                                    threshold=cfg.decision_threshold),
            "confusion": confusion(y_test, pred),
            "classification_report": report_dict(y_test, pred),
            "curves": curves(y_test, prob),
            "importance": model_importance(est),
            "permutation_importance": permutation_importance_table(
                est, X_test, y_test, n_repeats=cfg.permutation_repeats, seed=cfg.random_state),
            "model_file": path.name,
        }
        log.info("Test evaluation", extra={"fields": {
            "model": spec.key, "roc_auc": f"{model_results[spec.key]['test_metrics']['roc_auc']:.4f}",
            "accuracy": f"{model_results[spec.key]['test_metrics']['accuracy']:.4f}"}})

    # ---- Selected-model diagnostics (training data) -----------------------
    best_est = tuned[best_key]["search"].best_estimator_
    oof = cross_val_predict(clone(best_est), X_train, y_train, cv=inner,
                            method="predict_proba", n_jobs=cfg.n_jobs)[:, 1]
    sizes, lc_train, lc_val = learning_curve(
        clone(best_est), X_train, y_train, cv=inner, scoring=cfg.selection_metric,
        train_sizes=np.linspace(0.15, 1.0, 8), n_jobs=cfg.n_jobs, shuffle=True,
        random_state=cfg.random_state)
    comparisons = {k: paired_bootstrap_difference(y_test, test_probs[best_key], test_probs[k],
                                                  metric="roc_auc", n_boot=cfg.n_bootstrap,
                                                  seed=cfg.random_state)
                   for k in test_probs if k != best_key}

    shutil.copyfile(models_dir / f"{best_key}.joblib", models_dir / "best_model.joblib")
    model_hashes["best_model.joblib"] = file_sha256(models_dir / "best_model.joblib")

    preds = X_test.copy()
    preds.insert(0, "row_index", X_test.index)
    preds["survived"] = y_test.to_numpy()
    for k, p in test_probs.items():
        preds[f"prob_{k}"] = p
        preds[f"pred_{k}"] = (p >= cfg.decision_threshold).astype(int)
    preds.to_csv(reports_dir / config.TEST_PREDICTIONS_FILE.name, index=False)

    ranking = sorted(tuned, key=lambda k: tuned[k]["selection_mean"], reverse=True)
    runner_up = ranking[1] if len(ranking) > 1 else None
    results = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": asdict(cfg),
        "selection": {
            "metric": cfg.selection_metric,
            "estimate": "nested CV (outer folds)" if cfg.nested_cv else "inner CV of the best configuration",
            "rationale": config.SELECTION_METRIC_RATIONALE,
            "selected_model": best_key,
            "ranking": ranking,
            "runner_up": runner_up,
            "margin_over_runner_up": (tuned[best_key]["selection_mean"] - tuned[runner_up]["selection_mean"])
            if runner_up else None,
        },
        "data": {
            "source": config.DATA_SOURCE_URL,
            "sha256": file_sha256(data_path),
            "profile": profile,
            "n_train": int(len(X_train)), "n_test": int(len(X_test)),
            "train_positive_rate": float(y_train.mean()), "test_positive_rate": float(y_test.mean()),
            "features_used": config.MODEL_FEATURES,
            "feature_descriptions": FEATURE_DESCRIPTIONS,
            "excluded_features": config.EXCLUDED_FEATURES,
        },
        "models": model_results,
        "selected_diagnostics": {
            "oof_threshold_table": threshold_table(y_train, oof),
            "oof_metrics": compute_metrics(y_train, oof, cfg.decision_threshold),
            "learning_curve": {"train_sizes": sizes, "train_scores": lc_train, "val_scores": lc_val,
                               "metric": cfg.selection_metric},
            "paired_auc_vs_others": comparisons,
        },
        "environment": _environment(),
        "runtime_seconds": time.perf_counter() - started,
    }
    write_json(results, reports_dir / config.RESULTS_FILE.name)
    write_json({
        "generated_at": results["generated_at"],
        "selected_model": best_key,
        "selected_display_name": model_results[best_key]["display_name"],
        "decision_threshold": cfg.decision_threshold,
        "features": config.MODEL_FEATURES,
        "data_sha256": results["data"]["sha256"],
        "model_sha256": model_hashes,
        "environment": results["environment"],
    }, models_dir / config.METADATA_FILE.name)
    log.info("Experiment complete", extra={"fields": {
        "selected": best_key, "seconds": f"{results['runtime_seconds']:.1f}"}})
    return results


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train and evaluate Titanic survival models.")
    p.add_argument("--quick", action="store_true",
                   help="Fast run: no nested CV, 8 search iterations, 300 bootstrap samples.")
    p.add_argument("--seed", type=int, help="Random seed (default 42 or $TITANIC_SEED).")
    p.add_argument("--n-iter", type=int, help="Random-search iterations per model.")
    p.add_argument("--no-nested", action="store_true", help="Disable nested CV.")
    p.add_argument("--models", nargs="+", help="Subset of model keys to train.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    cfg = config.ExperimentConfig()
    if args.quick:
        cfg = replace(cfg, nested_cv=False, n_iter=8, n_bootstrap=300, permutation_repeats=10)
    if args.seed is not None:
        cfg = replace(cfg, random_state=args.seed)
    if args.n_iter is not None:
        cfg = replace(cfg, n_iter=args.n_iter)
    if args.no_nested:
        cfg = replace(cfg, nested_cv=False)
    config.ensure_directories()
    run_training(cfg, model_keys=args.models)


if __name__ == "__main__":
    main()
