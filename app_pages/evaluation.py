"""How well it works: test-set performance, in plain words first."""

import pandas as pd
import streamlit as st

from src.dashboard import state
from src.dashboard.explain import PLAIN_LABELS, detail_hint, detailed, ladder_summary, metric_sentence
from src.evaluation.metrics import METRIC_EXPLANATIONS, METRIC_LABELS, compute_metrics, confusion
from src.visualization import charts

st.title("How well it works")

res = state.require_results()
preds = state.predictions()
names = state.model_names(res)
sel_key = res["selection"]["selected_model"]
threshold = res["config"]["decision_threshold"]
n_test = res["data"]["n_test"]

st.markdown(f"After training, the model was tested **once** on {n_test} passengers it had never seen. "
            "Here is how it did.")

if detailed():
    key = st.segmented_control("Model", list(names),
                               format_func=lambda k: names[k] + (" ★" if k == sel_key else ""),
                               default=sel_key, required=True, key="ev_model")
else:
    key = sel_key
m = res["models"][key]
tm, ci, cm = m["test_metrics"], m["test_ci"], m["confusion"]

# ---- Plain answers --------------------------------------------------------------
with st.container(horizontal=True):
    for k in ("accuracy", "recall", "specificity", "precision"):
        label, tech = PLAIN_LABELS[k]
        st.metric(label, f"{tm[k]:.0%}", border=True, help=f"{metric_sentence(k, tm[k])} ({tech})")
st.markdown("\n".join(f"- {metric_sentence(k, tm[k], with_value=False)}"
                      for k in ("accuracy", "recall", "specificity", "precision")))

rows = state.ladder()
if rows is not None and key == sel_key:
    short, long = ladder_summary(rows)
    st.plotly_chart(charts.ladder_chart(rows), alt="Accuracy of simple rules compared with the model")
    st.markdown(short + " " + long)
    st.caption("Rules that need a choice (which groups 'survive') were worked out from the training "
               "passengers only — each group gets the outcome most of its members had — and then "
               "scored on the same test passengers as the model.")

left, right = st.columns(2, gap="large")
with left:
    st.plotly_chart(charts.confusion_matrix_fig(cm, f"Right and wrong answers ({n_test} test passengers)",
                                                plain=True),
                    alt="Confusion matrix")
with right:
    st.markdown("**Reading the four boxes**")
    st.markdown(
        f"- **{cm['tn']} + {cm['tp']} correct:** the model's answer matched what happened.\n"
        f"- **{cm['fn']} missed survivors:** survived, but the model said they would not.\n"
        f"- **{cm['fp']} false alarms:** died, but the model said they would survive.")
    worse = "missed survivors" if cm["fn"] > cm["fp"] else "false alarms"
    st.caption(f"The model's most common mistake is **{worse}**. The model turns its chance estimate "
               f"into a yes/no at {threshold:.0%}: above that it says 'survived'.")

@st.fragment
def threshold_explorer(preds: pd.DataFrame, key: str, default: float, base_cm: dict,
                       base_acc: float) -> None:
    """Slider that moves the yes/no cut-off; only this section reruns."""
    st.subheader("Try moving the cut-off")
    st.markdown("The model gives each passenger a chance of survival, then says *'survived'* if that chance "
                f"is above a cut-off ({default:.0%} by default). **Drag the slider** to see the trade-off: a "
                "lower cut-off finds more survivors but raises more false alarms; a higher one does the opposite.")
    t = st.slider("Cut-off", 0.05, 0.95, float(default), 0.05, format="%.2f", key="ev_threshold",
                  help="Explore only — the model in use keeps its 50% cut-off.")
    y, prob = preds["survived"].to_numpy(), preds[f"prob_{key}"].to_numpy()
    mt = compute_metrics(y, prob, t)
    cm_t = confusion(y, (prob >= t).astype(int))
    with st.container(horizontal=True):
        st.metric("Right overall", f"{mt['accuracy']:.0%}", int(round(100 * (mt["accuracy"] - base_acc))),
                  border=True, delta_description=f"points vs. {default:.0%} cut-off")
        st.metric("Survivors found", f"{cm_t['tp']} of {cm_t['tp'] + cm_t['fn']}",
                  int(cm_t["tp"] - base_cm["tp"]), border=True, delta_description=f"vs. {default:.0%} cut-off")
        st.metric("False alarms", cm_t["fp"], int(cm_t["fp"] - base_cm["fp"]), delta_color="inverse",
                  border=True, delta_description=f"vs. {default:.0%} cut-off")
    a, b = st.columns([3, 2], gap="large")
    a.plotly_chart(charts.probability_distribution(preds, key, t),
                   alt="Histogram of predicted chances of survival by actual outcome, with the cut-off")
    a.caption("Each bar counts test passengers by the chance the model gave them: people who died in "
              "red-orange, survivors in blue. Everyone right of the dashed line is predicted to survive.")
    b.plotly_chart(charts.confusion_matrix_fig(cm_t, f"Answers at cut-off {t:.0%}", plain=True),
                   alt="Confusion matrix at the chosen cut-off")
    st.caption("This is for exploring only. Choosing a cut-off by looking at the test results would make "
               "the grade above look better than it really is, so the model in use keeps its "
               f"{default:.0%} cut-off.")


if preds is not None:
    threshold_explorer(preds, key, threshold, cm, tm["accuracy"])

    p = preds.copy()
    p["correct"] = p[f"pred_{key}"] == p["survived"]
    errs = p[~p["correct"]]
    if len(errs):
        g = (errs.assign(kind=errs["survived"].map({1: "missed survivors", 0: "false alarms"}))
             .groupby(["kind", "sex", "pclass"]).size().sort_values(ascending=False).head(3))
        st.markdown("**Who it gets wrong most often:** " + "; ".join(
            f"{n} {kind} among {'women' if s == 'female' else 'men'} in class {c}" for (kind, s, c), n in g.items())
            + ". These are people whose fate went against the general pattern for their group, which "
              "the available details cannot explain.")

st.info(f"Because only {n_test} passengers were tested, the true accuracy is probably between "
        f"{ci['accuracy']['low']:.0%} and {ci['accuracy']['high']:.0%}. Small differences between "
        "models are within this uncertainty.", title="How sure are these numbers?",
        icon=":material/info:")

if not detailed():
    detail_hint()
    st.stop()

# ======================= Detailed mode =============================================
st.divider()
st.subheader("Technical evaluation")
with st.container(horizontal=True):
    for k in ("roc_auc", "average_precision", "accuracy", "balanced_accuracy", "f1", "mcc"):
        st.metric(METRIC_LABELS[k].split(" (")[0], f"{tm[k]:.3f}", border=True,
                  help=f"95% CI {ci[k]['low']:.3f}–{ci[k]['high']:.3f}. {METRIC_EXPLANATIONS[k]}")

tab_metrics, tab_curves, tab_cm, tab_thr, tab_err = st.tabs(
    ["Metrics", "ROC & PR curves", "Confusion matrix (%)", "Threshold analysis", "Errors"])

with tab_metrics:
    st.plotly_chart(charts.metric_comparison(res), alt="Grouped bar chart of test metrics with confidence intervals")
    st.caption("Taller is better. The thin lines show each score's 95% confidence interval; overlapping "
               "lines mean the models cannot be reliably told apart.")
    rows = [{"Metric": METRIC_LABELS[k], "Value": ci[k]["estimate"], "CI low": ci[k]["low"],
             "CI high": ci[k]["high"], "What it means": METRIC_EXPLANATIONS[k]} for k in ci]
    st.dataframe(pd.DataFrame(rows), hide_index=True, alt=f"Test metrics for {names[key]}",
                 column_config={c: st.column_config.NumberColumn(c, format="%.3f")
                                for c in ("Value", "CI low", "CI high")}
                 | {"What it means": st.column_config.TextColumn(width="large")})
    st.markdown("**Classification report**")
    st.dataframe(pd.DataFrame(m["classification_report"]).T.round(3), alt="Per-class precision, recall and F1")
    diffs = res["selected_diagnostics"]["paired_auc_vs_others"]
    if diffs:
        st.markdown(f"**Is {names[sel_key]} distinguishable from the others?** (paired bootstrap of ROC-AUC difference)")
        st.dataframe(pd.DataFrame([{"Compared with": names[k], "Δ ROC-AUC": d["difference"],
                                    "95% CI low": d["low"], "95% CI high": d["high"],
                                    "Excludes 0": d["low"] > 0 or d["high"] < 0,
                                    "Resamples where selected wins": d["share_a_better"]}
                                   for k, d in diffs.items()]).round(3),
                     hide_index=True, alt="Paired bootstrap comparisons")
    st.caption("Bootstrap CIs capture test-sample noise only, not variation from re-training on new data.")

with tab_curves:
    a, b = st.columns(2)
    a.plotly_chart(charts.roc_curves(res, highlight=key), alt="ROC curves for all models")
    b.plotly_chart(charts.pr_curves(res, highlight=key), alt="Precision-recall curves for all models")
    st.caption("Curves closer to the top-left (ROC) or top-right (PR) are better. ROC chance is the "
               "diagonal (AUC 0.5). PR chance is the survival rate in the test set, so PR-AUC should be "
               "compared with that baseline rather than 0.5.")

with tab_cm:
    st.plotly_chart(charts.confusion_matrix_fig(cm, f"{names[key]} — row percentages", normalize=True),
                    alt="Confusion matrix as row percentages")
    st.caption("Each row sums to 100%: the share of actual non-survivors (top) and survivors (bottom) "
               "that landed in each predicted column.")

with tab_thr:
    diag = res["selected_diagnostics"]
    st.markdown(f"Computed for **{names[sel_key]}** from **out-of-fold predictions on the training set**, "
                "so a threshold can be chosen without touching the test set. The deployed model keeps "
                f"threshold {threshold}.")
    st.plotly_chart(charts.threshold_curve(diag["oof_threshold_table"], current=threshold),
                    alt="Metrics as a function of decision threshold")
    st.caption("Moving the cut-off left flags more people as survivors: recall rises, precision falls.")
    st.dataframe(pd.DataFrame(diag["oof_threshold_table"])[
        ["threshold", "predicted_positive_rate", "precision", "recall", "specificity", "f1",
         "balanced_accuracy", "mcc"]].round(3), hide_index=True, alt="Threshold table")
    st.plotly_chart(charts.learning_curve_fig(diag["learning_curve"]), alt="Learning curve for the selected model")
    st.caption("A persistent gap between training and validation curves indicates variance (over-fitting); "
               "a validation curve still rising suggests more data would help. The OOF predictions reuse "
               "data that chose the hyperparameters, so they are mildly optimistic.")

with tab_err:
    if preds is None:
        st.info("Test predictions file is missing; re-run training.")
    else:
        p = preds.copy()
        p["correct"] = p[f"pred_{key}"] == p["survived"]
        p["outcome"] = p.apply(lambda r: ("TP" if r["survived"] else "FP") if r[f"pred_{key}"]
                               else ("FN" if r["survived"] else "TN"), axis=1)
        feat = st.segmented_control("Break errors down by", ["sex", "pclass", "embarked", "age", "fare"],
                                    default="sex", required=True, key="ev_err_feat")
        a, b = st.columns(2)
        a.plotly_chart(charts.correctness_by_feature(preds, key, feat), alt=f"Error rate by {feat}")
        with b:
            st.markdown(f"**{(~p['correct']).sum()} misclassified** of {len(p)} test passengers.")
            conf = p[~p["correct"]].assign(confidence=lambda d: (d[f"prob_{key}"] - 0.5).abs() + 0.5)
            st.dataframe(conf.sort_values("confidence", ascending=False)[
                ["outcome", f"prob_{key}", "pclass", "sex", "age", "fare", "sibsp", "parch", "embarked"]
            ].rename(columns={f"prob_{key}": "P(survived)"}).round(3).head(15), hide_index=True,
                alt="Most confident misclassifications")
            st.caption("Most confident mistakes first. FN = survivor predicted to die, "
                       "FP = non-survivor predicted to survive.")
