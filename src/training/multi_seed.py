"""Repeat the experiment over many random train/test splits.

The main experiment uses a single split (seed 42). With only 179 test
passengers, one split can favour or penalise a model by chance. This script
re-runs split -> tuning -> selection -> test evaluation for several seeds so
the spread of results is visible.

    python -m src.training.multi_seed               # 20 seeds, 10 candidates per model
    python -m src.training.multi_seed --seeds 5     # quicker

To keep the run to minutes, each seed uses a single (non-nested) 5-fold search;
the per-seed selection therefore uses the inner-CV score of the best candidate,
which is slightly optimistic but identical across models.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

import config
from src.data.loader import load_raw_data, split_features_target
from src.evaluation.metrics import compute_metrics
from src.training.models import get_model_specs
from src.training.train import _environment, select_best, split_data, tune_model
from src.utils import get_logger, write_json

log = get_logger("training.multi_seed")

MULTI_SEED_FILE: Path = config.REPORTS_DIR / "multi_seed_results.json"


def run_multi_seed(seeds: list[int], n_iter: int = 10, model_keys: list[str] | None = None,
                   out_path: Path = MULTI_SEED_FILE, n_jobs: int = -1) -> dict[str, Any]:
    """Run split + tuning + selection + test evaluation for each seed and save the results."""
    started = time.perf_counter()
    df = load_raw_data()
    X, y = split_features_target(df)
    specs = get_model_specs(model_keys)
    rows: list[dict[str, Any]] = []
    for i, seed in enumerate(seeds, 1):
        cfg = replace(config.ExperimentConfig(), random_state=seed, n_iter=n_iter, nested_cv=False,
                      n_jobs=n_jobs)
        X_tr, X_te, y_tr, y_te = split_data(X, y, cfg)
        tuned = {s.key: tune_model(s, X_tr, y_tr, cfg) for s in specs}
        winner = select_best(tuned)
        for s in specs:
            prob = tuned[s.key]["search"].best_estimator_.predict_proba(X_te)[:, 1]
            m = compute_metrics(y_te, prob, cfg.decision_threshold)
            rows.append({"seed": seed, "model": s.key, "display_name": s.display_name,
                         "cv_auc": tuned[s.key]["selection_mean"], "test_auc": m["roc_auc"],
                         "test_accuracy": m["accuracy"], "selected": s.key == winner})
        log.info("Seed done", extra={"fields": {"seed": seed, "progress": f"{i}/{len(seeds)}",
                                                "selected": winner}})

    table = pd.DataFrame(rows)
    summary = []
    for key, g in table.groupby("model", sort=False):
        summary.append({
            "model": key, "display_name": g["display_name"].iloc[0],
            "test_auc_mean": g["test_auc"].mean(), "test_auc_std": g["test_auc"].std(ddof=1),
            "test_auc_min": g["test_auc"].min(), "test_auc_max": g["test_auc"].max(),
            "test_acc_mean": g["test_accuracy"].mean(), "test_acc_std": g["test_accuracy"].std(ddof=1),
            "times_selected": int(g["selected"].sum()),
        })
    # How often the CV-selected model was also the best on that seed's test set.
    best_on_test = table.loc[table.groupby("seed")["test_auc"].idxmax(), ["seed", "model"]]
    chosen = table[table["selected"]][["seed", "model"]]
    agree = float((best_on_test.set_index("seed")["model"] == chosen.set_index("seed")["model"]).mean())
    results = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seeds": list(seeds), "n_iter": n_iter, "nested_cv": False,
        "config": asdict(replace(config.ExperimentConfig(), n_iter=n_iter, nested_cv=False)),
        "rows": rows,
        "summary": summary,
        "selected_was_test_best_share": agree,
        "selected_test_auc_mean": float(table[table["selected"]]["test_auc"].mean()),
        "environment": _environment(),
        "runtime_seconds": time.perf_counter() - started,
    }
    write_json(results, out_path)
    log.info("Multi-seed complete", extra={"fields": {"seeds": len(seeds),
                                                      "seconds": f"{results['runtime_seconds']:.0f}"}})
    return results


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Repeat the experiment over several random splits.")
    p.add_argument("--seeds", type=int, default=20, help="Number of seeds (default 20).")
    p.add_argument("--first-seed", type=int, default=42,
                   help="First seed; seeds are consecutive, so the default includes the main run's seed 42.")
    p.add_argument("--n-iter", type=int, default=10, help="Search candidates per model (default 10).")
    p.add_argument("--models", nargs="+", help="Subset of model keys.")
    args = p.parse_args(argv)
    config.ensure_directories()
    run_multi_seed(list(range(args.first_seed, args.first_seed + args.seeds)), n_iter=args.n_iter,
                   model_keys=args.models)


if __name__ == "__main__":
    main()
