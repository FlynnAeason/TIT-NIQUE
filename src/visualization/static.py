"""Static Matplotlib figures (PNG) for the PDF report.

Plotly static export would require the extra `kaleido` dependency, so the PDF
uses Matplotlib renderings of the same artifacts with the same palette.
"""

from __future__ import annotations

import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import config  # noqa: E402
from src.evaluation.metrics import METRIC_LABELS  # noqa: E402

INK, MUTED = "#1F2937", "#6B7280"
LINESTYLES = ["-", "--", ":", "-."]

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.edgecolor": "#D1D5DB", "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": "#E5E7EB", "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "axes.axisbelow": True,
})


def _png(fig) -> bytes:
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=200)
    plt.close(fig)
    return buf.getvalue()


def _color(i: int) -> str:
    return config.MODEL_COLORS[i % len(config.MODEL_COLORS)]


def roc_pr_png(results: dict) -> bytes:
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 3.3))
    a.plot([0, 1], [0, 1], color=MUTED, ls=":", lw=1)
    prev = None
    for i, m in enumerate(results["models"].values()):
        r, p = m["curves"]["roc"], m["curves"]["pr"]
        prev = m["curves"]["prevalence"]
        a.plot(r["fpr"], r["tpr"], color=_color(i), ls=LINESTYLES[i % 4], lw=1.8,
               label=f"{m['display_name']} ({m['test_metrics']['roc_auc']:.3f})")
        b.step(p["recall"], p["precision"], where="post", color=_color(i), ls=LINESTYLES[i % 4], lw=1.8,
               label=f"{m['display_name']} ({m['test_metrics']['average_precision']:.3f})")
    b.axhline(prev, color=MUTED, ls=":", lw=1)
    a.set(title="ROC (AUC)", xlabel="False positive rate", ylabel="True positive rate", xlim=(0, 1), ylim=(0, 1.02))
    b.set(title="Precision-recall (AP)", xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1.02))
    a.legend(fontsize=7, loc="lower right")
    b.legend(fontsize=7, loc="lower left")
    return _png(fig)


def confusion_png(cm: dict, title: str) -> bytes:
    z = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
    fig, ax = plt.subplots(figsize=(3.4, 2.9))
    ax.imshow(z, cmap="Blues", vmin=0)
    ax.grid(False)
    labels = [["TN", "FP"], ["FN", "TP"]]
    for r in range(2):
        for c in range(2):
            ax.text(c, r, f"{labels[r][c]}\n{z[r, c]}", ha="center", va="center",
                    color="white" if z[r, c] > z.max() * 0.6 else INK, fontsize=10)
    ax.set_xticks([0, 1], ["Did not survive", "Survived"])
    ax.set_yticks([0, 1], ["Did not survive", "Survived"])
    ax.set(xlabel="Predicted", ylabel="Actual", title=title)
    return _png(fig)


def metric_comparison_png(results: dict) -> bytes:
    metrics = ["roc_auc", "average_precision", "accuracy", "balanced_accuracy", "f1", "mcc"]
    models = list(results["models"].values())
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    width = 0.8 / len(models)
    x = np.arange(len(metrics))
    for i, m in enumerate(models):
        est = np.array([m["test_ci"][k]["estimate"] for k in metrics])
        lo = np.array([m["test_ci"][k]["low"] for k in metrics])
        hi = np.array([m["test_ci"][k]["high"] for k in metrics])
        ax.bar(x + i * width - 0.4 + width / 2, est, width, color=_color(i), edgecolor="white",
               label=m["display_name"], yerr=[est - lo, hi - est], ecolor=MUTED, capsize=0, error_kw={"lw": 0.8})
    labels = [METRIC_LABELS[k].split(" (")[0] for k in metrics]
    ax.set_xticks(x, [lb.replace(" ", "\n", 1) if len(lb) > 10 else lb for lb in labels])
    ax.set(ylim=(0, 1), ylabel="Test score", title="Test metrics with bootstrap 95% CIs")
    ax.legend(fontsize=7, ncol=len(models), loc="upper center", bbox_to_anchor=(0.5, -0.12))
    return _png(fig)


def importance_png(values: dict, value_col: str, title: str, error_col: str | None = None,
                   top: int = 12) -> bytes:
    df = pd.DataFrame(values).head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.4, 0.6 + 0.24 * len(df)))
    colors = ([config.PALETTE["blue"] if v > 0 else config.PALETTE["vermillion"] for v in df[value_col]]
              if value_col == "coefficient" else config.PALETTE["blue"])
    ax.barh(df["feature"], df[value_col], color=colors,
            xerr=df[error_col] if error_col else None, ecolor=MUTED, error_kw={"lw": 0.8})
    if value_col == "coefficient":
        ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_title(title)
    return _png(fig)


def survival_rates_png(profile: dict) -> bytes:
    keys = [k for k in ("sex", "pclass", "embarked") if k in profile["survival_by"]]
    fig, axes = plt.subplots(1, len(keys), figsize=(7.2, 2.5), sharey=True)
    for ax, k in zip(np.atleast_1d(axes), keys):
        t = pd.DataFrame(profile["survival_by"][k])
        rate = t["survival_rate"].to_numpy()
        ax.bar(t[k].astype(str), rate, color=config.PALETTE["blue"], width=0.6,
               yerr=[rate - t["ci_low"], t["ci_high"] - rate], ecolor=MUTED, error_kw={"lw": 0.8})
        ax.set_title(f"by {k}")
        ax.set_ylim(0, 1)
    np.atleast_1d(axes)[0].set_ylabel("Survival rate")
    return _png(fig)
