"""Training workflow, persistence, prediction and reporting (end-to-end)."""

import io
import json

import numpy as np

import pandas as pd
import pytest

import config
from src.data.loader import split_features_target
from src.prediction.predict import (
    DataValidationError,
    ModelUnavailableError,
    load_model,
    predict_frame,
    predict_one,
    read_batch_csv,
)
from src.reporting.pdf_report import build_pdf
from src.reporting.summary import build_summary, comparison_table, summary_markdown
from src.training.train import select_best


def test_training_writes_artifacts(trained):
    results, dirs = trained
    assert (dirs["models_dir"] / "best_model.joblib").exists()
    assert (dirs["models_dir"] / "metadata.json").exists()
    assert (dirs["reports_dir"] / "experiment_results.json").exists()
    assert set(results["models"]) == {"logistic_regression", "decision_tree"}
    json.loads((dirs["reports_dir"] / "experiment_results.json").read_text())  # strict JSON


def test_selection_uses_cv_not_test(trained):
    results, _ = trained
    sel = results["selection"]["selected_model"]
    scores = {k: m["selection_mean"] for k, m in results["models"].items()}
    assert sel == max(scores, key=scores.get)


def test_select_best_tie_breaks_on_variance():
    s = {"a": {"selection_mean": 0.85, "selection_std": 0.05},
         "b": {"selection_mean": 0.85, "selection_std": 0.01}}
    assert select_best(s) == "b"


def test_test_set_untouched_by_tuning(trained, raw_df):
    results, dirs = trained
    split = pd.read_csv(dirs["processed_dir"] / "split_indices.csv")
    n_test = (split["split"] == "test").sum()
    assert n_test == results["data"]["n_test"]
    # Search CV folds only cover the training rows.
    assert results["data"]["n_train"] + n_test == len(raw_df)


def test_metrics_in_valid_ranges(trained):
    results, _ = trained
    for m in results["models"].values():
        for k, v in m["test_metrics"].items():
            if k in ("mcc",):
                assert -1 <= v <= 1
            elif k != "log_loss":
                assert 0 <= v <= 1, k
        cm = m["confusion"]
        assert sum(cm.values()) == results["data"]["n_test"]


def test_model_round_trip_and_hash_check(trained, raw_df, passenger):
    _, dirs = trained
    model = load_model(models_dir=dirs["models_dir"])
    out = predict_one(model, passenger)
    assert out["prediction"] in (config.POSITIVE_LABEL, config.NEGATIVE_LABEL)
    assert out["prob_survived"] + out["prob_not_survived"] == pytest.approx(1.0, abs=1e-3)

    # Re-loading a specific model gives identical probabilities.
    again = load_model("best_model.joblib", models_dir=dirs["models_dir"])
    X = raw_df[config.MODEL_FEATURES].head(20)
    assert (model.pipeline.predict_proba(X) == again.pipeline.predict_proba(X)).all()


def test_tampered_model_is_rejected(trained, tmp_path):
    _, dirs = trained
    import shutil
    target = tmp_path / "models"
    shutil.copytree(dirs["models_dir"], target)
    with (target / "best_model.joblib").open("ab") as fh:
        fh.write(b"tamper")
    with pytest.raises(ModelUnavailableError, match="Hash check failed"):
        load_model(models_dir=target)


def test_path_traversal_rejected(trained):
    _, dirs = trained
    with pytest.raises(ModelUnavailableError):
        load_model("../reports/experiment_results.json", models_dir=dirs["models_dir"])


def test_missing_model_reports_clearly(tmp_path):
    with pytest.raises(ModelUnavailableError):
        load_model(models_dir=tmp_path)


def test_batch_prediction(trained, raw_df):
    _, dirs = trained
    model = load_model(models_dir=dirs["models_dir"])
    csv = raw_df[config.MODEL_FEATURES + ["who"]].head(25).to_csv(index=False)
    out = predict_frame(model, read_batch_csv(io.StringIO(csv)))
    assert len(out) == 25 and "who" in out.columns
    assert out["prob_survived"].between(0, 1).all()


def test_batch_prediction_rejects_invalid_rows(trained):
    _, dirs = trained
    model = load_model(models_dir=dirs["models_dir"])
    bad = "pclass,sex,age,sibsp,parch,fare,embarked\n1,female,30,0,0,50,S\n5,alien,30,0,0,50,Z\n"
    with pytest.raises(DataValidationError) as exc:
        predict_frame(model, read_batch_csv(io.StringIO(bad)))
    assert any("rows 2" in e for e in exc.value.errors)


def test_what_if_curves(trained, passenger):
    from src.prediction.whatif import GRIDS, biggest_swing, what_if_curve
    _, dirs = trained
    model = load_model(models_dir=dirs["models_dir"])
    for feature, grid in GRIDS.items():
        curve = what_if_curve(model, passenger, feature)
        assert len(curve) == len(grid)
        assert curve["prob"].between(0, 1).all()
    swings = biggest_swing(model, passenger)
    sizes = [hi - lo for _, lo, hi in swings]
    assert sizes == sorted(sizes, reverse=True)
    # Sex is the strongest signal, so changing only sex must change the output.
    sex_curve = what_if_curve(model, passenger, "sex")
    assert sex_curve["prob"].nunique() == 2


def test_baseline_ladder(raw_df):
    from src.evaluation.baselines import baseline_ladder
    from src.training.train import split_data
    X, y = split_features_target(raw_df)
    _, X_te, _, y_te = split_data(X, y, config.ExperimentConfig())
    fake = {"accuracy": 0.8, "recall": 0.7, "specificity": 0.85}
    rows = baseline_ladder(raw_df, X_te.index, "Test model", fake)
    assert rows[0]["accuracy"] == pytest.approx(1 - y_te.mean())  # everyone dies
    assert rows[-1]["is_model"] and rows[-1]["accuracy"] == 0.8
    assert all(r["n"] == len(y_te) for r in rows)
    # The sex rule is learned from training data: women are the majority survivors there.
    assert "female" in rows[1]["detail"]


def test_baseline_rules_ignore_test_labels(raw_df):
    """Flipping test outcomes must not change which groups a rule predicts to survive."""
    from src.evaluation.baselines import baseline_ladder
    from src.training.train import split_data
    X, y = split_features_target(raw_df)
    _, X_te, _, _ = split_data(X, y, config.ExperimentConfig())
    flipped = raw_df.copy()
    flipped.loc[X_te.index, config.TARGET] = 1 - flipped.loc[X_te.index, config.TARGET]
    fake = {"accuracy": 0.8, "recall": 0.7, "specificity": 0.85}
    a = baseline_ladder(raw_df, X_te.index, "m", fake)
    b = baseline_ladder(flipped, X_te.index, "m", fake)
    assert [r["detail"] for r in a] == [r["detail"] for r in b]


def test_model_vote(trained, passenger):
    from src.prediction.trust import model_vote
    _, dirs = trained
    models = {k: load_model(f"{k}.joblib", models_dir=dirs["models_dir"])
              for k in ("logistic_regression", "decision_tree")}
    v = model_vote(models, passenger)
    assert v["n"] == 2 and len(v["votes"]) == 2
    assert v["level"] in ("unanimous", "split")
    assert 0 <= v["low"] <= v["high"] <= 1


def test_similar_passengers(raw_df):
    from src.prediction.trust import similar_passengers
    p = {"pclass": 1, "sex": "female", "age": 30.0, "sibsp": 0, "parch": 0, "fare": 80, "embarked": "S"}
    s = similar_passengers(raw_df, p)
    assert s.n >= 10
    assert (s.matches["sex"] == "female").all() and (s.matches["pclass"] == 1).all()
    assert s.survived == int(s.matches["survived"].sum())
    assert s.ci[0] <= s.rate <= s.ci[1]


def test_similar_passengers_unknown_age_and_fallback(raw_df):
    from src.prediction.trust import similar_passengers
    p = {"pclass": 2, "sex": "male", "age": None, "sibsp": 0, "parch": 0, "fare": 13, "embarked": "S"}
    s = similar_passengers(raw_df, p)
    assert s.n > 0 and "men" in s.criteria
    # An age far outside the data forces the fallback to sex + class only.
    old = similar_passengers(raw_df, {**p, "age": 95.0})
    assert old.criteria == ["men", "in 2nd class"]


def test_probability_grid(trained, raw_df, passenger):
    from src.prediction.landscape import AXES, matching_passengers, probability_grid
    from src.prediction.whatif import what_if_curve
    _, dirs = trained
    model = load_model(models_dir=dirs["models_dir"])
    xs, ys, z = probability_grid(model, passenger, "age", "fare")
    assert z.shape == (len(AXES["fare"][1]), len(AXES["age"][1]))
    assert ((z >= 0) & (z <= 1)).all()
    # One row of the grid must equal the one-detail what-if curve at that fare.
    row = list(ys).index(100.0)
    curve = what_if_curve(model, {**passenger, "fare": 100.0}, "age")
    assert np.allclose(z[row], np.interp(xs, curve["value"], curve["prob"]), atol=1e-3)
    with pytest.raises(ValueError):
        probability_grid(model, passenger, "age", "age")
    people = matching_passengers(raw_df, passenger, "age", "fare")
    assert set(people.columns) == {"x", "y", "survived"} and people.notna().all(axis=None)


def test_multi_seed_run(tmp_path, raw_df):
    from src.training.multi_seed import run_multi_seed
    out = tmp_path / "ms.json"
    res = run_multi_seed([1, 2], n_iter=1, model_keys=["logistic_regression", "decision_tree"],
                         out_path=out, n_jobs=1)
    assert out.exists()
    assert len(res["rows"]) == 4
    assert sum(r["selected"] for r in res["rows"]) == 2  # one winner per seed
    assert {s["model"] for s in res["summary"]} == {"logistic_regression", "decision_tree"}
    assert 0 <= res["selected_was_test_best_share"] <= 1


def test_summary_uses_actual_results(trained):
    results, _ = trained
    sel = results["models"][results["selection"]["selected_model"]]
    text = summary_markdown(build_summary(results))
    assert sel["display_name"] in text
    assert f"{sel['test_metrics']['roc_auc']:.3f}" in text
    assert len(comparison_table(results)) == len(results["models"])


def test_pdf_report_generation(trained):
    results, _ = trained
    pdf = build_pdf(results)
    assert pdf.startswith(b"%PDF") and len(pdf) > 50_000
    assert pdf.count(b"/Type /Image") + pdf.count(b"/Subtype /Image") >= 4
