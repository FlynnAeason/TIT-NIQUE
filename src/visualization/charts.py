"""Interactive Plotly figures built from real data and experiment artifacts.

Colour use follows one rule per job: model identity uses the fixed Okabe-Ito
order in ``config.MODEL_COLORS`` (plus a dash pattern as secondary encoding);
class identity uses ``config.CLASS_COLORS``; magnitude uses a single-hue
sequential scale. All figures share :func:`_style`.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

import config
from src.evaluation.metrics import METRIC_LABELS

FONT = "Inter, Segoe UI, Helvetica, Arial, sans-serif"
INK = "#1F2937"
MUTED = "#6B7280"
GRID = "rgba(107,114,128,0.18)"
SEQUENTIAL = [[0, "#EFF6FB"], [0.5, "#56B4E9"], [1, "#0B3C5D"]]
DASHES = ["solid", "dash", "dot", "dashdot", "longdash", "longdashdot"]
CLASS_NAMES = {0: config.NEGATIVE_LABEL, 1: config.POSITIVE_LABEL}


def _style(fig: go.Figure, title: str, height: int = 380, **layout: Any) -> go.Figure:
    legend = {"orientation": "h", "yanchor": "bottom", "y": 1.0, "xanchor": "right", "x": 1,
              "title": None} | layout.pop("legend", {})
    margin = {"l": 10, "r": 10, "t": 50, "b": 10} | layout.pop("margin", {})
    fig.update_layout(
        # Text colours are left to the host theme (Streamlit light/dark) so labels
        # stay readable in both modes.
        title={"text": title, "x": 0, "xanchor": "left", "font": {"size": 15}},
        font={"family": FONT, "size": 12},
        height=height,
        margin=margin,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        hoverlabel={"font": {"family": FONT}},
        legend=legend,
        **layout,
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False, linecolor=GRID, automargin=True)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor=GRID, automargin=True)
    return fig


def model_color(i: int) -> str:
    return config.MODEL_COLORS[i % len(config.MODEL_COLORS)]


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
def target_distribution(df: pd.DataFrame) -> go.Figure:
    counts = df[config.TARGET].value_counts().reindex([0, 1]).fillna(0)
    total = counts.sum()
    fig = go.Figure(go.Bar(
        x=[CLASS_NAMES[i] for i in counts.index], y=counts.values,
        marker_color=[config.CLASS_COLORS[i] for i in counts.index],
        text=[f"{v:,.0f} ({v / total:.1%})" for v in counts.values], textposition="outside",
        hovertemplate="%{x}: %{y} passengers<extra></extra>", width=0.5))
    fig.update_yaxes(title="Passengers", rangemode="tozero")
    fig.update_traces(cliponaxis=False)
    return _style(fig, "Survival outcome", height=320, showlegend=False)


def survival_rate_bar(table: pd.DataFrame, column: str, label: str | None = None) -> go.Figure:
    """Survival rate per level with Wilson 95% CI error bars."""
    t = table.copy()
    t[column] = t[column].astype(str)
    fig = go.Figure(go.Bar(
        x=t[column], y=t["survival_rate"], marker_color=config.PALETTE["blue"],
        error_y={"type": "data", "symmetric": False, "array": t["ci_high"] - t["survival_rate"],
                 "arrayminus": t["survival_rate"] - t["ci_low"], "color": MUTED, "thickness": 1.2},
        customdata=np.c_[t["count"], t["survivors"], t["ci_low"], t["ci_high"]],
        hovertemplate=("%{x}<br>Survival rate %{y:.1%}<br>95% CI %{customdata[2]:.1%}–"
                       "%{customdata[3]:.1%}<br>%{customdata[1]} of %{customdata[0]} survived<extra></extra>"),
        width=0.55))
    fig.update_yaxes(title="Survival rate", tickformat=".0%", range=[0, 1])
    fig.update_xaxes(title=label or column, type="category")
    return _style(fig, f"Survival rate by {label or column} (95% Wilson CI)", height=330, showlegend=False)


def numeric_distribution(df: pd.DataFrame, column: str, log_x: bool = False,
                         bins: int = 30) -> go.Figure:
    """Overlaid histograms of a numeric feature split by outcome."""
    fig = go.Figure()
    data = df[[column, config.TARGET]].dropna()
    values = np.log1p(data[column]) if log_x else data[column]
    for cls in (0, 1):
        v = values[data[config.TARGET] == cls]
        fig.add_trace(go.Histogram(
            x=v, name=CLASS_NAMES[cls], nbinsx=bins, opacity=0.65,
            marker={"color": config.CLASS_COLORS[cls]},
            hovertemplate=f"{CLASS_NAMES[cls]}<br>{column}: %{{x}}<br>Passengers: %{{y}}<extra></extra>"))
    unit = {"age": " (years)", "fare": " (pre-1970 GBP)"}.get(column, "")
    fig.update_layout(barmode="overlay", bargap=0.06)
    fig.update_xaxes(title=(f"log(1 + {column})" if log_x else column) + unit)
    fig.update_yaxes(title="Passengers")
    n_missing = int(df[column].isna().sum())
    note = f" — {n_missing} missing excluded" if n_missing else ""
    return _style(fig, f"Distribution of {column} by outcome{note}", height=340)


def box_by_target(df: pd.DataFrame, column: str) -> go.Figure:
    fig = go.Figure()
    for cls in (0, 1):
        fig.add_trace(go.Box(y=df.loc[df[config.TARGET] == cls, column], name=CLASS_NAMES[cls],
                             marker_color=config.CLASS_COLORS[cls], boxpoints="outliers"))
    fig.update_yaxes(title=column)
    return _style(fig, f"{column}: spread and outliers by outcome", height=340, showlegend=False)


def missing_values_bar(missing: pd.DataFrame) -> go.Figure:
    m = missing[missing["missing"] > 0].sort_values("percent")
    fig = go.Figure(go.Bar(
        x=m["percent"], y=m["column"], orientation="h", marker_color=config.PALETTE["blue"],
        text=[f"{p:.1f}%" for p in m["percent"]], textposition="outside",
        customdata=m["missing"], hovertemplate="%{y}: %{customdata} missing (%{x:.1f}%)<extra></extra>"))
    fig.update_xaxes(title="Missing (%)", range=[0, 100])
    fig.update_traces(cliponaxis=False)
    return _style(fig, "Missing values by column", height=60 + 40 * max(len(m), 1), showlegend=False)


def association_heatmap(mat: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Heatmap(
        z=mat.values, x=mat.columns, y=mat.index, zmin=0, zmax=1, colorscale=SEQUENTIAL,
        text=mat.round(2).values, texttemplate="%{text}", textfont={"size": 11},
        hovertemplate="%{y} × %{x}<br>association %{z:.3f}<extra></extra>",
        colorbar={"title": "Strength", "thickness": 12}))
    fig.update_yaxes(autorange="reversed")
    return _style(fig, "Feature association (|Spearman| or Cramér's V)", height=460)


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
def _model_items(results: dict) -> list[tuple[int, str, dict]]:
    return [(i, k, m) for i, (k, m) in enumerate(results["models"].items())]


def cv_score_distribution(results: dict, metric: str = "roc_auc") -> go.Figure:
    """Per-fold selection scores (nested CV when available) for each model."""
    fig = go.Figure()
    for i, key, m in _model_items(results):
        scores = (m["nested_cv"] or m["inner_cv_best"])[metric]
        fig.add_trace(go.Box(
            y=scores, name=m["display_name"], boxpoints="all", jitter=0.3, pointpos=0,
            marker={"color": model_color(i), "size": 8}, line={"color": model_color(i)},
            hovertemplate=f"{m['display_name']}<br>fold {METRIC_LABELS[metric]}: %{{y:.3f}}<extra></extra>"))
    source = "nested CV outer folds" if results["config"]["nested_cv"] else "inner CV folds"
    fig.update_yaxes(title=METRIC_LABELS[metric])
    return _style(fig, f"Cross-validation {METRIC_LABELS[metric]} by fold ({source})", showlegend=False)


def metric_comparison(results: dict, metrics: list[str] | None = None) -> go.Figure:
    """Grouped bars of test metrics with bootstrap 95% CIs."""
    metrics = metrics or ["roc_auc", "average_precision", "accuracy", "balanced_accuracy", "f1", "mcc"]
    fig = go.Figure()
    for i, key, m in _model_items(results):
        est = [m["test_ci"][x]["estimate"] for x in metrics]
        lo = [m["test_ci"][x]["low"] for x in metrics]
        hi = [m["test_ci"][x]["high"] for x in metrics]
        fig.add_trace(go.Bar(
            name=m["display_name"], x=[METRIC_LABELS[x] for x in metrics], y=est,
            marker={"color": model_color(i)},
            error_y={"type": "data", "symmetric": False, "array": np.subtract(hi, est),
                     "arrayminus": np.subtract(est, lo), "color": MUTED, "thickness": 1},
            customdata=np.c_[lo, hi],
            hovertemplate=(f"{m['display_name']}<br>%{{x}}: %{{y:.3f}}"
                           "<br>95% CI %{customdata[0]:.3f}–%{customdata[1]:.3f}<extra></extra>")))
    fig.update_layout(barmode="group", bargap=0.25, bargroupgap=0.08)
    fig.update_yaxes(title="Score (test set)", range=[0, 1])
    return _style(fig, "Test-set metrics with bootstrap 95% CIs", height=420)


def roc_curves(results: dict, highlight: str | None = None) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Chance (AUC 0.5)",
                             line={"color": MUTED, "dash": "dot", "width": 1}, hoverinfo="skip"))
    for i, key, m in _model_items(results):
        c = m["curves"]["roc"]
        auc = m["test_metrics"]["roc_auc"]
        fig.add_trace(go.Scatter(
            x=c["fpr"], y=c["tpr"], mode="lines", name=f"{m['display_name']} (AUC {auc:.3f})",
            line={"color": model_color(i), "width": 3 if key == highlight else 2, "dash": DASHES[i % 6]},
            customdata=c["threshold"],
            hovertemplate=(f"{m['display_name']}<br>FPR %{{x:.3f}}<br>TPR %{{y:.3f}}"
                           "<br>threshold %{customdata:.2f}<extra></extra>")))
    fig.update_xaxes(title="False positive rate (1 − specificity)", range=[0, 1])
    fig.update_yaxes(title="True positive rate (recall)", range=[0, 1.02])
    return _style(fig, "ROC curves (test set)", height=440,
                  legend={"orientation": "v", "x": 0.98, "y": 0.02, "xanchor": "right", "yanchor": "bottom"})


def pr_curves(results: dict, highlight: str | None = None) -> go.Figure:
    fig = go.Figure()
    prevalence = next(iter(results["models"].values()))["curves"]["prevalence"]
    fig.add_trace(go.Scatter(x=[0, 1], y=[prevalence, prevalence], mode="lines",
                             name=f"Chance (AP {prevalence:.2f})",
                             line={"color": MUTED, "dash": "dot", "width": 1}, hoverinfo="skip"))
    for i, key, m in _model_items(results):
        c = m["curves"]["pr"]
        ap = m["test_metrics"]["average_precision"]
        fig.add_trace(go.Scatter(
            x=c["recall"], y=c["precision"], mode="lines", name=f"{m['display_name']} (AP {ap:.3f})",
            line={"color": model_color(i), "width": 3 if key == highlight else 2,
                  "dash": DASHES[i % 6], "shape": "hv"},
            hovertemplate=f"{m['display_name']}<br>Recall %{{x:.3f}}<br>Precision %{{y:.3f}}<extra></extra>"))
    fig.update_xaxes(title="Recall", range=[0, 1])
    fig.update_yaxes(title="Precision", range=[0, 1.02])
    return _style(fig, "Precision–recall curves (test set)", height=440,
                  legend={"orientation": "v", "x": 0.02, "y": 0.02, "xanchor": "left", "yanchor": "bottom"})


def confusion_matrix_fig(cm: dict[str, int], title: str, normalize: bool = False,
                         plain: bool = False) -> go.Figure:
    """2x2 confusion matrix. ``plain=True`` uses everyday wording instead of TP/FP terms."""
    z = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]], dtype=float)
    shown = z / z.sum(axis=1, keepdims=True) if normalize else z
    if plain:
        labels = [["Correct ✓", "False alarm"], ["Missed survivor", "Correct ✓"]]
        x = ["Model said: died", "Model said: survived"]
        y = ["Actually died", "Actually survived"]
    else:
        labels = [["True negative", "False positive"], ["False negative", "True positive"]]
        x = ["Pred: did not survive", "Pred: survived"]
        y = ["Actual: did not survive", "Actual: survived"]
    text = [[f"{labels[r][c]}<br><b>{shown[r, c]:.1%}</b>" if normalize else
             f"{labels[r][c]}<br><b>{int(z[r, c])}</b>" for c in range(2)] for r in range(2)]
    fig = go.Figure(go.Heatmap(
        z=shown, x=x, y=y, colorscale=SEQUENTIAL, showscale=False,
        text=text, texttemplate="%{text}", customdata=z,
        hovertemplate="%{y}<br>%{x}<br>count %{customdata:.0f}<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    fig.update_traces(textfont={"size": 13})
    return _style(fig, title, height=330)


# Diverging scale for a chance of survival: vermillion (low) -> neutral grey at 50% -> blue (high).
CHANCE_SCALE = [[0.0, config.CLASS_COLORS[0]], [0.5, "#E5E7EB"], [1.0, config.CLASS_COLORS[1]]]


def landscape_3d(xs: np.ndarray, ys: np.ndarray, z: np.ndarray, x_label: str, y_label: str,
                 threshold: float, passengers: pd.DataFrame | None = None,
                 jitter_x: float = 0.0, jitter_y: float = 0.0) -> go.Figure:
    """Chance of survival as a 3D surface with the cut-off plane and real passengers.

    Real passengers sit above the surface (survived) or below it (died); for
    whole-number axes a small seeded jitter keeps overlapping points visible.
    """
    fig = go.Figure()
    fig.add_trace(go.Surface(
        x=xs, y=ys, z=z, colorscale=CHANCE_SCALE, cmin=0, cmax=1, name="Model's chance",
        colorbar={"title": {"text": "Chance of survival"}, "tickformat": ".0%", "thickness": 14, "len": 0.7},
        lighting={"ambient": 0.9, "diffuse": 0.5, "specular": 0.1},
        hovertemplate=f"{x_label}: %{{x}}<br>{y_label}: %{{y}}<br>Chance of survival: %{{z:.0%}}<extra></extra>"))
    fig.add_trace(go.Surface(
        x=xs, y=ys, z=np.full_like(z, threshold, dtype=float), showscale=False, opacity=0.22,
        colorscale=[[0, MUTED], [1, MUTED]], hoverinfo="skip", name=f"Cut-off {threshold:.0%}"))
    if passengers is not None and len(passengers):
        rng = np.random.default_rng(0)
        for outcome, z_pos, label in ((1, 1.04, "Real passengers who survived"),
                                      (0, -0.04, "Real passengers who died")):
            g = passengers[passengers["survived"] == outcome]
            if g.empty:
                continue
            fig.add_trace(go.Scatter3d(
                x=g["x"] + rng.uniform(-jitter_x, jitter_x, len(g)) if jitter_x else g["x"],
                y=g["y"] + rng.uniform(-jitter_y, jitter_y, len(g)) if jitter_y else g["y"],
                z=np.full(len(g), z_pos), mode="markers", name=label,
                marker={"size": 3.2, "color": config.CLASS_COLORS[outcome], "opacity": 0.85,
                        "line": {"width": 0}},
                hovertemplate=f"{label}<br>{x_label}: %{{x:.0f}}<br>{y_label}: %{{y:.0f}}<extra></extra>"))
    axis = {"gridcolor": GRID, "zerolinecolor": GRID, "backgroundcolor": "rgba(0,0,0,0)", "showspikes": False}
    fig.update_layout(scene={
        "xaxis": {**axis, "title": {"text": x_label}},
        "yaxis": {**axis, "title": {"text": y_label}},
        "zaxis": {**axis, "title": {"text": "Chance of survival"}, "range": [-0.08, 1.08], "tickformat": ".0%"},
        "camera": {"eye": {"x": 1.45, "y": -1.45, "z": 0.75}, "center": {"x": 0, "y": 0, "z": -0.12}},
        "aspectmode": "manual", "aspectratio": {"x": 1.3, "y": 1.3, "z": 0.75},
        "domain": {"x": [0, 1], "y": [0, 1]},
    })
    return _style(fig, "The model's chance of survival, as a landscape", height=560,
                  legend={"orientation": "h", "y": 0.0, "yanchor": "top", "x": 0, "xanchor": "left"},
                  margin={"l": 0, "r": 0, "t": 40, "b": 30})


def landscape_flat(xs: np.ndarray, ys: np.ndarray, z: np.ndarray, x_label: str, y_label: str,
                   threshold: float) -> go.Figure:
    """The same landscape seen from directly above, with a line where the answer flips."""
    fig = go.Figure(go.Heatmap(
        x=xs, y=ys, z=z, colorscale=CHANCE_SCALE, zmin=0, zmax=1,
        colorbar={"title": {"text": "Chance"}, "tickformat": ".0%", "thickness": 14},
        hovertemplate=f"{x_label}: %{{x}}<br>{y_label}: %{{y}}<br>Chance of survival: %{{z:.0%}}<extra></extra>"))
    fig.add_trace(go.Contour(
        x=xs, y=ys, z=z, showscale=False, hoverinfo="skip", contours={"start": threshold, "end": threshold,
                                                                       "size": 1, "coloring": "none"},
        line={"color": "#111827", "width": 2.5}, name=f"Cut-off {threshold:.0%}"))
    fig.update_xaxes(title=x_label)
    fig.update_yaxes(title=y_label)
    return _style(fig, "Seen from above: the black line is where the answer flips", height=380,
                  showlegend=False)


def ladder_chart(rows: list[dict]) -> go.Figure:
    """Accuracy of each rung of the baseline ladder; the model's bar is highlighted."""
    labels = [r["step"] for r in rows][::-1]
    acc = [r["accuracy"] for r in rows][::-1]
    colors = [config.PALETTE["blue"] if r["is_model"] else config.PALETTE["grey"] for r in rows][::-1]
    text = [f"{r['accuracy']:.0%} · {r['correct']} of {r['n']}" for r in rows][::-1]
    fig = go.Figure(go.Bar(x=acc, y=labels, orientation="h", marker_color=colors, text=text,
                           textposition="outside", cliponaxis=False,
                           hovertemplate="%{y}<br>right on %{x:.1%} of test passengers<extra></extra>"))
    fig.update_xaxes(title="Share of test passengers predicted correctly", range=[0, 1.12], tickformat=".0%")
    return _style(fig, "How much does the model add over simple rules?", height=300, showlegend=False)


def multi_seed_chart(ms: dict, original_seed: int | None = None) -> go.Figure:
    """Test ROC-AUC per model across random splits; the original split is highlighted."""
    rows = pd.DataFrame(ms["rows"])
    order = (rows.groupby("display_name")["test_auc"].mean().sort_values(ascending=False).index.tolist())
    # Colour follows the model (registration order, as in every other chart), not its rank here.
    color_of = {s["display_name"]: model_color(i) for i, s in enumerate(ms["summary"])}
    fig = go.Figure()
    for name in order:
        g = rows[rows["display_name"] == name]
        c = color_of.get(name, MUTED)
        fig.add_trace(go.Box(
            y=g["test_auc"], name=name, boxpoints="all", jitter=0.35, pointpos=0,
            marker={"color": c, "size": 7, "opacity": 0.75}, line={"color": c},
            customdata=g["seed"], hovertemplate=f"{name}<br>split %{{customdata}}: %{{y:.3f}}<extra></extra>"))
        if original_seed is not None:
            o = g[g["seed"] == original_seed]
            if len(o):
                fig.add_trace(go.Scatter(
                    x=[name], y=o["test_auc"], mode="markers", showlegend=False,
                    marker={"symbol": "diamond", "size": 13, "color": "rgba(0,0,0,0)",
                            "line": {"color": MUTED, "width": 2}},
                    hovertemplate=f"{name}<br>original split: %{{y:.3f}}<extra></extra>"))
    fig.update_yaxes(title="Ranking skill on the test set (ROC-AUC)")
    return _style(fig, f"Test scores across {len(ms['seeds'])} different random splits", height=400,
                  showlegend=False)


def what_if_chart(curve: pd.DataFrame, label: str, current: Any, threshold: float,
                  categorical: bool, format_value=str) -> go.Figure:
    """Chance of survival as one detail changes, with the current value highlighted."""
    fig = go.Figure()
    if categorical:
        names = [format_value(v) for v in curve["value"]]
        fig.add_trace(go.Bar(
            x=names, y=curve["prob"], width=0.55,
            marker_color=[config.PALETTE["blue"] if v == current else config.PALETTE["grey"]
                          for v in curve["value"]],
            text=[f"{p:.0%}" for p in curve["prob"]], textposition="outside", cliponaxis=False,
            hovertemplate=f"{label}: %{{x}}<br>Chance of survival: %{{y:.0%}}<extra></extra>"))
        fig.update_xaxes(title=label, type="category")
    else:
        fig.add_trace(go.Scatter(
            x=curve["value"], y=curve["prob"], mode="lines", line={"color": config.PALETTE["blue"], "width": 2.5},
            hovertemplate=f"{label}: %{{x}}<br>Chance of survival: %{{y:.0%}}<extra></extra>"))
        if current is not None:
            y_now = float(np.interp(float(current), curve["value"].astype(float), curve["prob"]))
            fig.add_trace(go.Scatter(
                x=[current], y=[y_now], mode="markers", marker={"size": 12, "color": config.PALETTE["orange"]},
                hovertemplate="Your passenger<br>%{y:.0%}<extra></extra>"))
        fig.update_xaxes(title=label)
    fig.add_hline(y=threshold, line={"color": MUTED, "dash": "dash", "width": 1.5},
                  annotation_text=f"cut-off {threshold:.0%}", annotation_position="top left")
    fig.update_yaxes(title="Chance of survival", range=[0, 1.08], tickformat=".0%")
    return _style(fig, f"What if only {label.lower()} changed?", height=340, showlegend=False)


def probability_distribution(preds: pd.DataFrame, key: str, threshold: float = 0.5) -> go.Figure:
    fig = go.Figure()
    for cls in (0, 1):
        fig.add_trace(go.Histogram(
            x=preds.loc[preds["survived"] == cls, f"prob_{key}"], name=f"Actual: {CLASS_NAMES[cls]}",
            xbins={"start": 0, "end": 1.0001, "size": 0.05}, opacity=0.65,
            marker={"color": config.CLASS_COLORS[cls]},
            hovertemplate="P(survived) %{x}<br>Passengers %{y}<extra></extra>"))
    fig.add_vline(x=threshold, line={"color": MUTED, "dash": "dash", "width": 1.5},
                  annotation_text=f"cut-off {threshold:.0%}", annotation_position="top")
    fig.update_layout(barmode="overlay", bargap=0.06)
    fig.update_xaxes(title="Model's chance of survival", range=[0, 1], tickformat=".0%")
    fig.update_yaxes(title="Test passengers")
    # Legend below the plot so it never collides with the cut-off label at the top.
    return _style(fig, "Model's chance of survival vs. what actually happened (test set)", height=380,
                  legend={"orientation": "h", "y": -0.28, "yanchor": "top", "x": 0, "xanchor": "left"})


def threshold_curve(table: list[dict], metrics: list[str] | None = None,
                    current: float = 0.5) -> go.Figure:
    metrics = metrics or ["precision", "recall", "f1", "balanced_accuracy"]
    df = pd.DataFrame(table)
    fig = go.Figure()
    for i, m in enumerate(metrics):
        fig.add_trace(go.Scatter(x=df["threshold"], y=df[m], mode="lines+markers", name=METRIC_LABELS[m],
                                 line={"color": model_color(i), "width": 2, "dash": DASHES[i % 6]},
                                 marker={"size": 6},
                                 hovertemplate=f"{METRIC_LABELS[m]}<br>threshold %{{x:.2f}}: %{{y:.3f}}<extra></extra>"))
    fig.add_vline(x=current, line={"color": MUTED, "dash": "dash", "width": 1.5})
    fig.update_xaxes(title="Decision threshold on P(survived)", range=[0, 1])
    fig.update_yaxes(title="Score", range=[0, 1.02])
    return _style(fig, "Threshold trade-offs (out-of-fold training predictions)", height=400)


def learning_curve_fig(lc: dict) -> go.Figure:
    sizes = np.asarray(lc["train_sizes"])
    fig = go.Figure()
    for name, scores, color, dash in (("Training", lc["train_scores"], config.PALETTE["orange"], "dash"),
                                      ("Validation (CV)", lc["val_scores"], config.PALETTE["blue"], "solid")):
        arr = np.asarray(scores, dtype=float)
        mean, std = arr.mean(axis=1), arr.std(axis=1)
        fig.add_trace(go.Scatter(x=np.r_[sizes, sizes[::-1]], y=np.r_[mean + std, (mean - std)[::-1]],
                                 fill="toself", fillcolor=color, opacity=0.15, line={"width": 0},
                                 hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=sizes, y=mean, mode="lines+markers", name=name,
                                 line={"color": color, "width": 2, "dash": dash}, marker={"size": 7},
                                 hovertemplate=f"{name}<br>%{{x}} rows: %{{y:.3f}}<extra></extra>"))
    fig.update_xaxes(title="Training rows")
    fig.update_yaxes(title=METRIC_LABELS.get(lc["metric"], lc["metric"]))
    return _style(fig, "Learning curve (selected model, mean ± 1 SD)", height=380)


def importance_bar(values: dict, value_col: str, title: str, error_col: str | None = None,
                   top: int = 15, x_title: str = "Importance") -> go.Figure:
    df = pd.DataFrame(values).head(top).iloc[::-1]
    signed = value_col == "coefficient"
    colors = ([config.PALETTE["blue"] if v > 0 else config.PALETTE["vermillion"] for v in df[value_col]]
              if signed else config.PALETTE["blue"])
    fig = go.Figure(go.Bar(
        x=df[value_col], y=df["feature"], orientation="h", marker_color=colors,
        error_x={"type": "data", "array": df[error_col], "color": MUTED, "thickness": 1} if error_col else None,
        hovertemplate="%{y}: %{x:.4f}<extra></extra>"))
    if signed:
        fig.add_vline(x=0, line={"color": MUTED, "width": 1})
    fig.update_xaxes(title=x_title)
    return _style(fig, title, height=120 + 26 * len(df), showlegend=False)


def correctness_by_feature(preds: pd.DataFrame, key: str, column: str) -> go.Figure:
    """Error rate of a model on the test set, split by a feature."""
    d = preds.copy()
    d["correct"] = d[f"pred_{key}"] == d["survived"]
    if pd.api.types.is_numeric_dtype(d[column]) and d[column].nunique() > 6:
        d[column] = pd.qcut(d[column], 4, duplicates="drop").astype(str)
    d[column] = d[column].astype(str).replace("nan", "missing")
    g = d.groupby(column)["correct"].agg(["mean", "count"]).reset_index()
    fig = go.Figure(go.Bar(x=g[column], y=1 - g["mean"], marker_color=config.PALETTE["vermillion"],
                           customdata=g["count"],
                           hovertemplate="%{x}<br>error rate %{y:.1%}<br>n = %{customdata}<extra></extra>",
                           width=0.55))
    fig.update_yaxes(title="Error rate", tickformat=".0%", rangemode="tozero")
    fig.update_xaxes(title=column, type="category")
    return _style(fig, f"Test error rate by {column}", height=320, showlegend=False)
