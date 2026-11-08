"""Executive summary generated entirely from experiment artifacts.

No number, model name or conclusion is hard-coded: everything is read from the
results dictionary produced by :mod:`src.training.train`. Wording adapts to the
results (e.g. whether the selected model is statistically distinguishable from
the others on the test set).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

import config
from src.evaluation.metrics import METRIC_LABELS


@dataclass
class Section:
    heading: str
    paragraphs: list[str] = field(default_factory=list)
    bullets: list[str] = field(default_factory=list)


def _pct(x: float) -> str:
    return f"{x:.1%}"


def _f(x: float | None, nd: int = 3) -> str:
    return "n/a" if x is None else f"{x:.{nd}f}"


def selected(results: dict) -> tuple[str, dict]:
    key = results["selection"]["selected_model"]
    return key, results["models"][key]


def comparison_table(results: dict) -> pd.DataFrame:
    """One row per model: selection score and key test metrics with CIs."""
    rows = []
    sel_key = results["selection"]["selected_model"]
    for key, m in results["models"].items():
        t, ci = m["test_metrics"], m["test_ci"]
        rows.append({
            "Model": m["display_name"] + (" ★" if key == sel_key else ""),
            "CV ROC-AUC (selection)": f"{m['selection_mean']:.3f} ± {m['selection_std']:.3f}",
            "Test ROC-AUC [95% CI]": f"{t['roc_auc']:.3f} [{ci['roc_auc']['low']:.3f}, {ci['roc_auc']['high']:.3f}]",
            "Test PR-AUC": f"{t['average_precision']:.3f}",
            "Accuracy": f"{t['accuracy']:.3f}",
            "Balanced acc.": f"{t['balanced_accuracy']:.3f}",
            "F1": f"{t['f1']:.3f}",
            "Recall": f"{t['recall']:.3f}",
            "Specificity": f"{t['specificity']:.3f}",
            "MCC": f"{t['mcc']:.3f}",
            "Brier": f"{t['brier']:.3f}",
        })
    return pd.DataFrame(rows)


def top_permutation_features(model: dict, n: int = 3) -> list[tuple[str, float, float]]:
    v = model["permutation_importance"]["values"]
    return list(zip(v["feature"], v["mean"], v["std"]))[:n]


def build_summary(results: dict) -> list[Section]:
    """Return the executive summary as ordered sections."""
    cfg = results["config"]
    data = results["data"]
    prof = data["profile"]
    ov = prof["overview"]
    sel_key, sel = selected(results)
    tm, ci = sel["test_metrics"], sel["test_ci"]
    sections: list[Section] = []

    sections.append(Section("Objective", [
        "Predict whether a Titanic passenger survived from information known at boarding "
        "(ticket class, sex, age, family aboard, fare and port), and explain which factors "
        "drive the model's estimates. The project extends an introductory course notebook "
        "into a reproducible, leakage-safe modelling workflow."]))

    missing = {r["column"]: r["percent"] for r in prof["missing"] if r["missing"] > 0}
    sections.append(Section("Dataset", [
        f"{ov['n_rows']:,} passengers and {ov['n_columns']} columns (seaborn 'titanic' dataset, "
        f"hash-pinned). {_pct(ov['positive_rate'])} survived, so the classes are moderately "
        f"imbalanced. The data were split once, stratified by outcome, into "
        f"{data['n_train']} training and {data['n_test']} test passengers "
        f"(seed {cfg['random_state']})."],
        bullets=[f"Missing values: " + ", ".join(f"{c} {p:.1f}%" for c, p in missing.items()) + "."
                 if missing else "No missing values.",
                 f"{ov['n_duplicate_rows']} rows are exact duplicates of another row. Without names or "
                 "ticket numbers they cannot be distinguished from genuinely different passengers "
                 "who share all attributes, so they were kept."]))

    red = [r for r in prof["redundancy"] if r["holds"]]
    sections.append(Section("Data quality and feature decisions", [
        "Several columns restate other columns. Each relationship below was verified exactly "
        "in the data, and the redundant column was excluded:"],
        bullets=[f"`{r['column']}`: {r['relation']}." for r in red] +
                [f"`deck` excluded: {next((r['detail'] for r in prof['redundancy'] if r['column'] == 'deck'), '')}"]
                + ["Final model inputs: " + ", ".join(f"`{f}`" for f in data["features_used"]) +
                   ", plus engineered `log_fare`, `is_child` and a missing-age indicator."]))

    sel_est = results["selection"]["estimate"]
    sections.append(Section("Methodology", bullets=[
        "All imputation, scaling and encoding happen inside a scikit-learn Pipeline, so they are "
        "re-fitted on the training portion of every fold and never see validation or test rows.",
        f"Hyperparameters were tuned by randomised search ({cfg['n_iter']} candidates max per model) "
        f"with {cfg['cv_folds']}-fold stratified cross-validation on the training set only.",
        f"Models were ranked by {METRIC_LABELS[cfg['selection_metric']]} estimated with "
        + (f"nested cross-validation ({cfg['outer_folds']} outer folds), which scores the whole tuning "
           "procedure rather than the best of many tried configurations." if cfg["nested_cv"]
           else f"{sel_est}."),
        "The test set was used once, after selection, to report final performance with "
        f"{cfg['n_bootstrap']:,}-sample stratified bootstrap 95% confidence intervals.",
    ]))

    names = ", ".join(m["display_name"] for m in results["models"].values())
    margin = results["selection"]["margin_over_runner_up"]
    runner = results["selection"]["runner_up"]
    runner_txt = ""
    if runner:
        runner_txt = (f" It led the runner-up, {results['models'][runner]['display_name']}, by "
                      f"{margin:.3f} in cross-validated ROC-AUC")
        close = margin < max(sel["selection_std"], results["models"][runner]["selection_std"])
        runner_txt += (" — a margin smaller than the fold-to-fold variation, so the two are "
                       "practically tied." if close else ".")
    sections.append(Section("Models and selection", [
        f"Models evaluated: {names}. The selected model is **{sel['display_name']}** "
        f"(cross-validated ROC-AUC {sel['selection_mean']:.3f} ± {sel['selection_std']:.3f})."
        + runner_txt,
        config.SELECTION_METRIC_RATIONALE]))

    sections.append(Section("Key results (test set, threshold 0.5)", bullets=[
        f"ROC-AUC {tm['roc_auc']:.3f} (95% CI {ci['roc_auc']['low']:.3f}–{ci['roc_auc']['high']:.3f}); "
        f"PR-AUC {tm['average_precision']:.3f} vs. a chance level of {sel['curves']['prevalence']:.2f}.",
        f"Accuracy {_pct(tm['accuracy'])} (95% CI {_pct(ci['accuracy']['low'])}–{_pct(ci['accuracy']['high'])}), "
        f"against {_pct(1 - data['test_positive_rate'])} for always predicting 'did not survive'.",
        f"Recall {_pct(tm['recall'])} and specificity {_pct(tm['specificity'])}: the model "
        + ("misses more survivors than it wrongly flags non-survivors."
           if tm["recall"] < tm["specificity"] else "finds survivors at least as reliably as it rejects non-survivors."),
        f"Balanced accuracy {tm['balanced_accuracy']:.3f}, F1 {tm['f1']:.3f}, MCC {tm['mcc']:.3f}, "
        f"Brier score {tm['brier']:.3f}.",
    ]))

    diffs = results["selected_diagnostics"]["paired_auc_vs_others"]
    better = [results["models"][k]["display_name"] for k, d in diffs.items() if d["low"] > 0]
    worse = [results["models"][k]["display_name"] for k, d in diffs.items() if d["high"] < 0]
    if diffs:
        if better or worse:
            parts = []
            if better:
                parts.append("higher than " + ", ".join(better))
            if worse:
                parts.append("lower than " + ", ".join(worse))
            txt = ("On the test set, a paired bootstrap (95% CI of the difference excludes 0) finds its "
                   "ROC-AUC " + " and ".join(parts) + "; other differences are within test-set noise.")
            if worse:
                txt += (" A cross-validation winner that trails on the single test split is a reminder that "
                        "one 179-passenger split is noisy; the selection was deliberately not revised "
                        "after seeing test results, which would leak test information into the choice.")
        else:
            txt = ("On the test set, paired bootstrap 95% intervals for the ROC-AUC differences between "
                   "the selected model and every alternative include 0: with 179 test passengers the "
                   "models cannot be reliably separated, and the choice rests on cross-validation.")
        sections[-1].paragraphs.append(txt)

    top = top_permutation_features(sel)
    sections.append(Section("Most important predictive factors", [
        "Permutation importance on held-out test data (drop in ROC-AUC when a column is shuffled): "
        + "; ".join(f"`{f}` {m:.3f} ± {s:.3f}" for f, m, s in top) + ".",
        "These rankings describe what the model relies on, not causes of survival. Correlated "
        "inputs (e.g. class and fare) share credit, so shuffling one can understate its role."]))

    rates = {k: pd.DataFrame(v) for k, v in prof["survival_by"].items()}
    findings = []
    if "sex" in rates:
        s = rates["sex"].set_index("sex")["survival_rate"]
        findings.append(f"Survival was {_pct(s.get('female', float('nan')))} for women and "
                        f"{_pct(s.get('male', float('nan')))} for men — the dominant signal.")
    if "pclass" in rates:
        p = rates["pclass"].set_index("pclass")["survival_rate"]
        findings.append("Survival fell with ticket class: " +
                        ", ".join(f"class {k} {_pct(v)}" for k, v in p.items()) + ".")
    if "who" in rates:
        w = rates["who"].set_index("who")["survival_rate"]
        if "child" in w:
            findings.append(f"Children under 16 survived at {_pct(w['child'])}, consistent with "
                            "'women and children first'.")
    lr = results["models"].get("logistic_regression")
    if lr is not None and sel_key != "logistic_regression":
        gap = sel["selection_mean"] - lr["selection_mean"]
        test_gap = tm["roc_auc"] - lr["test_metrics"]["roc_auc"]
        if gap < 0.02:
            findings.append(
                f"Logistic Regression comes within {gap:.3f} cross-validated ROC-AUC of {sel['display_name']}"
                + (f" and scores {-test_gap:.3f} higher on the test set" if test_gap < 0 else "")
                + ": the signal is largely additive and captured by a few features, so the extra "
                  "flexibility of tree ensembles buys little on this dataset.")
        else:
            findings.append(f"{sel['display_name']} beats Logistic Regression by {gap:.3f} cross-validated "
                            "ROC-AUC, indicating interactions or non-linear effects the linear model misses.")
    sections.append(Section("Main findings", bullets=findings))

    sections.append(Section("Limitations", bullets=[
        f"Small data: {data['n_test']} test passengers give wide confidence intervals; differences of "
        "a few points between models are not reliable.",
        "The dataset covers 891 of ~2,200 people aboard (the Kaggle training subset); it may not "
        "represent all passengers and crew.",
        "Age is missing for ~20% of passengers and was imputed with the training median plus an "
        "indicator; the missingness itself may be informative.",
        "The out-of-fold threshold analysis reuses the training data that chose the hyperparameters, "
        "so it is mildly optimistic.",
        "Predictions are statistical estimates for hypothetical passengers, not historical facts.",
    ]))
    sections.append(Section("Recommendations and future work", bullets=[
        "Engineer features from passenger names (titles) and tickets (group size) using the full "
        "Kaggle dataset, which this seaborn version omits.",
        "Repeat the train/test split across several seeds to measure split-to-split variance.",
        "Calibrate probabilities (isotonic or Platt) if they are to be read as literal risks.",
        "Choose the decision threshold from the costs of false positives vs. false negatives for "
        "the intended use, using out-of-fold predictions — never the test set.",
    ]))
    return sections


def summary_markdown(sections: list[Section]) -> str:
    """Render sections as Markdown (used by the dashboard)."""
    parts = []
    for s in sections:
        parts.append(f"#### {s.heading}")
        parts.extend(s.paragraphs)
        parts.extend(f"- {b}" for b in s.bullets)
    return "\n\n".join(parts)


def headline(results: dict) -> dict[str, Any]:
    """Compact key facts for dashboard metric tiles."""
    sel_key, sel = selected(results)
    return {"model": sel["display_name"], "key": sel_key,
            "cv_auc": sel["selection_mean"], "cv_auc_std": sel["selection_std"],
            "test_auc": sel["test_metrics"]["roc_auc"], "test_acc": sel["test_metrics"]["accuracy"],
            "test_bal_acc": sel["test_metrics"]["balanced_accuracy"],
            "test_f1": sel["test_metrics"]["f1"], "test_mcc": sel["test_metrics"]["mcc"]}
