"""How it was chosen: the model contest, plain first, technical in Detailed mode."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import config
from src.dashboard import state
from src.dashboard.explain import detail_hint, detailed
from src.evaluation.metrics import METRIC_LABELS
from src.reporting.summary import comparison_table
from src.visualization import charts

st.title("How it was chosen")

res = state.require_results()
cfg = res["config"]
sel_key = res["selection"]["selected_model"]
names = state.model_names(res)
sel = res["models"][sel_key]

PLAIN_MODEL = {
    "logistic_regression": "adds up points for each detail (e.g. + for female, − for 3rd class)",
    "random_forest": "hundreds of decision trees that vote",
    "decision_tree": "a single flowchart of yes/no questions",
    "hist_gradient_boosting": "many small trees, each fixing the previous ones' mistakes",
    "svm": "draws the widest possible boundary between survivors and non-survivors",
}

st.markdown(f"We trained **{len(names)} different kinds of model** and held a fair contest. Each model "
            "was graded several times on different slices of the training passengers "
            "(*cross-validation*) — the test passengers were kept locked away until the winner was picked, "
            "so the final grade is honest.")
st.markdown("\n".join(f"- **{names[k]}** — {PLAIN_MODEL.get(k, '')}" for k in names))

order = sorted(res["models"], key=lambda k: res["models"][k]["selection_mean"])
fig = go.Figure(go.Bar(
    x=[res["models"][k]["selection_mean"] for k in order], y=[names[k] for k in order], orientation="h",
    marker_color=[config.PALETTE["blue"] if k == sel_key else config.PALETTE["grey"] for k in order],
    text=[f"{res['models'][k]['selection_mean']:.3f}" for k in order], textposition="outside",
    error_x={"type": "data", "array": [res["models"][k]["selection_std"] for k in order], "thickness": 1},
    hovertemplate="%{y}: %{x:.3f}<extra></extra>"))
fig.update_xaxes(range=[0.5, 1], title="Ranking skill during training (ROC-AUC; 0.5 = coin flip)")
fig.update_layout(height=260, margin={"l": 10, "r": 40, "t": 40, "b": 10}, showlegend=False,
                  title={"text": "Contest scores (winner in blue)", "x": 0})
st.plotly_chart(fig, alt="Cross-validated score per model, winner highlighted")

margin = res["selection"]["margin_over_runner_up"]
runner = res["selection"]["runner_up"]
if runner is not None:
    close = margin < max(sel["selection_std"], res["models"][runner]["selection_std"])
    st.markdown(f"**{names[sel_key]} won**" + (
        f", but only just: its lead over {names[runner]} ({margin:.3f}) is smaller than the normal "
        "wobble between grading rounds (the thin lines), so the two are effectively tied."
        if close else f", ahead of {names[runner]} by {margin:.3f}."))
diffs = res["selected_diagnostics"]["paired_auc_vs_others"]
beat_on_test = [names[k] for k, d in diffs.items() if d["high"] < 0]
if beat_on_test:
    st.info(f"On the final test, {', '.join(beat_on_test)} actually ranked passengers better than "
            f"{names[sel_key]}. We kept the original choice on purpose: switching to whichever model "
            "happens to do best on the test would turn the test into part of the contest, and the "
            f"reported grade would no longer be honest. With only {res['data']['n_test']} test "
            "passengers, a gap this size can come from which passengers landed in the test.",
            title="A twist at the end", icon=":material/info:")
st.subheader("Was it a fluke? The same contest on many different splits")
ms = state.multi_seed()
if ms is None:
    st.info("Not run yet. This check repeats the whole contest with different random choices of test "
            "passengers to see how much the results depend on luck. It takes about 10 minutes:",
            icon=":material/hourglass_empty:")
    st.code("python -m src.training.multi_seed", language="bash")
else:
    summ = pd.DataFrame(ms["summary"]).sort_values("test_auc_mean", ascending=False)
    n = len(ms["seeds"])
    st.markdown(f"We re-ran the whole contest **{n} times**, each time picking a different random 20% of "
                "passengers as the test group. Each dot below is one run.")
    orig_seed = cfg["random_state"] if cfg["random_state"] in ms["seeds"] else None
    st.plotly_chart(charts.multi_seed_chart(ms, orig_seed), alt="Test scores per model across many random splits")
    st.caption("Higher is better. Each box holds the middle half of the runs."
               + (" The hollow diamond marks the original split used everywhere else in this app."
                  if orig_seed is not None else ""))

    top, second = summ.iloc[0], summ.iloc[1]
    typical_wobble = summ["test_auc_std"].mean()
    lines = [f"**Average test score:** " + ", ".join(
        f"{r.display_name} {r.test_auc_mean:.3f}" for r in summ.itertuples()) + "."]
    if top.test_auc_mean - second.test_auc_mean < typical_wobble:
        lines.append(f"**No clear winner.** The gap between the top two ({top.test_auc_mean - second.test_auc_mean:.3f}) "
                     f"is smaller than how much one model's score moves from split to split "
                     f"(about ±{typical_wobble:.3f}). Which passengers end up in the test group matters "
                     "more than which of the leading models you pick.")
    else:
        lines.append(f"**{top.display_name} is consistently ahead**, by more than the usual split-to-split "
                     f"wobble (about ±{typical_wobble:.3f}).")
    if orig_seed is not None:
        rows = pd.DataFrame(ms["rows"])
        mine = rows[rows["model"] == sel_key]
        val = mine.loc[mine["seed"] == orig_seed, "test_auc"].iloc[0]
        rank = (mine["test_auc"] < val).mean()
        where = "one of its weakest" if rank <= 0.25 else "one of its best" if rank >= 0.75 else "a typical"
        lines.append(f"**On the original split, {names[sel_key]} scored {val:.3f} — {where} results** "
                     f"(its range: {mine['test_auc'].min():.3f}–{mine['test_auc'].max():.3f}). "
                     + ("So its weaker test score earlier on this page is likely, at least in part, down "
                        "to which passengers happened to land in that test group." if rank <= 0.25 else ""))
    wins = ", ".join(f"{r.display_name} {r.times_selected}×" for r in
                     pd.DataFrame(ms["summary"]).sort_values("times_selected", ascending=False).itertuples()
                     if r.times_selected)
    lines.append(f"**How often each model won the contest:** {wins}. The contest's pick was also the best "
                 f"on that run's test group {ms['selected_was_test_best_share']:.0%} of the time"
                 + (" — when models are this close, which one comes out on top in a single test group "
                    "is largely chance." if ms["selected_was_test_best_share"] < 0.5 else "."))
    st.markdown("\n".join(f"- {line}" for line in lines))
    st.caption(f"To keep this to minutes, each run uses a quicker search ({ms['n_iter']} candidates per "
               "model, no nested cross-validation) than the main experiment.")

st.caption("The contest score uses ROC-AUC because it measures how well a model *ranks* survivors above "
           "non-survivors, without depending on the 50% cut-off.")

if not detailed():
    detail_hint()
    st.stop()

# ======================= Detailed mode =============================================
st.divider()
with st.container(border=True):
    a, b = st.columns([3, 2])
    a.markdown(f"**Selection rule:** highest {METRIC_LABELS[cfg['selection_metric']]} estimated by "
               f"*{res['selection']['estimate']}*; ties broken by lower variance. "
               f"Selected: **{names[sel_key]}**.")
    a.caption(res["selection"]["rationale"])
    b.markdown(
        f"- Split: {res['data']['n_train']} train / {res['data']['n_test']} test (stratified)\n"
        f"- Inner CV: {cfg['cv_folds']}-fold stratified, seed {cfg['random_state']}\n"
        f"- Search: randomised, ≤ {cfg['n_iter']} candidates per model\n"
        + (f"- Outer CV: {cfg['outer_folds']} folds (nested)" if cfg["nested_cv"] else "- Nested CV: off"))

metric = st.segmented_control("Cross-validation metric", ["roc_auc", "average_precision", "accuracy",
                                                           "balanced_accuracy", "f1", "mcc"],
                              format_func=lambda m: METRIC_LABELS[m], default="roc_auc", required=True,
                              key="lab_metric")
left, right = st.columns([3, 2])
with left:
    st.plotly_chart(charts.cv_score_distribution(res, metric), alt="Per-fold cross-validation scores by model")
    st.caption("Each dot is one grading round; the box shows the middle half of the scores.")
with right:
    rows = []
    for k, m in res["models"].items():
        scores = (m["nested_cv"] or m["inner_cv_best"])[metric]
        rows.append({"Model": names[k], "Mean": sum(scores) / len(scores),
                     "Std": pd.Series(scores).std(), "Min": min(scores), "Max": max(scores),
                     "Tuning time (s)": m["tuning_seconds"]})
    st.dataframe(pd.DataFrame(rows).sort_values("Mean", ascending=False).round(3), hide_index=True,
                 alt="Cross-validation summary per model")
    if cfg["nested_cv"]:
        st.caption("Nested CV re-runs the whole search inside each outer fold, so the score reflects "
                   "the tuning procedure, not the luckiest of many configurations.")

st.subheader("Model configurations")
key = st.segmented_control("Model", list(names), format_func=names.get, default=sel_key, required=True,
                           key="lab_model")
m = res["models"][key]
c1, c2 = st.columns(2)
with c1:
    with st.container(border=True):
        st.markdown(f"**{m['display_name']}**" + ("  · selected" if key == sel_key else ""))
        st.caption(m["description"])
        st.markdown("**Search space**")
        st.dataframe(pd.DataFrame({"Hyperparameter": list(m["search_space"]),
                                   "Values / distribution": [str(v) for v in m["search_space"].values()]}),
                     hide_index=True, alt="Hyperparameter search space")
with c2:
    with st.container(border=True):
        st.markdown(f"**Best configuration** ({m['n_candidates']} candidates evaluated)")
        st.dataframe(pd.DataFrame({"Hyperparameter": list(m["best_params"]),
                                   "Value": [str(v) for v in m["best_params"].values()]}),
                     hide_index=True, alt="Tuned hyperparameters")
        st.caption(f"Inner-CV ROC-AUC of this configuration: {pd.Series(m['inner_cv_best']['roc_auc']).mean():.3f} — "
                   "optimistic, because it is the best of the candidates tried.")

st.markdown("**Search results** (top candidates by inner-CV ROC-AUC)")
table = pd.DataFrame(m["search_table"])
st.dataframe(table, hide_index=True, alt="Randomised search results",
             column_config={c: st.column_config.NumberColumn(c, format="%.4f")
                            for c in table.columns if c.startswith(("mean_", "std_", "fit_"))})

st.subheader("Comparison and selection")
st.dataframe(comparison_table(res), hide_index=True, alt="Model comparison table")
st.caption("★ marks the selected model. Test columns are shown for reference after selection; "
           "they were not used to choose the model.")
