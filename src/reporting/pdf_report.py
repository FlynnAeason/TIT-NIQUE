"""PDF report generated with ReportLab from the current experiment artifacts."""

from __future__ import annotations

import io
import re
from datetime import datetime
from pathlib import Path
from xml.sax.saxutils import escape

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image,
    KeepTogether,
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

import config
from src.evaluation.metrics import METRIC_EXPLANATIONS, METRIC_LABELS
from src.reporting.summary import build_summary, comparison_table, selected
from src.utils import get_logger
from src.visualization import static

log = get_logger(__name__)

PAGE_W, PAGE_H = A4
MARGIN = 18 * mm
CONTENT_W = PAGE_W - 2 * MARGIN
ACCENT = colors.HexColor(config.PALETTE["blue"])
INK = colors.HexColor("#1F2937")
MUTED = colors.HexColor("#6B7280")
RULE = colors.HexColor("#E5E7EB")


class ReportGenerationError(RuntimeError):
    """Raised when the PDF cannot be produced."""


def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    body = ParagraphStyle("body", parent=base["BodyText"], fontName="Helvetica", fontSize=9.5,
                          leading=13.5, textColor=INK, alignment=TA_LEFT, spaceAfter=5)
    return {
        "title": ParagraphStyle("title", parent=body, fontName="Helvetica-Bold", fontSize=22,
                                leading=27, spaceAfter=4),
        "subtitle": ParagraphStyle("subtitle", parent=body, fontSize=11, textColor=MUTED, spaceAfter=14),
        "h1": ParagraphStyle("h1", parent=body, fontName="Helvetica-Bold", fontSize=14, leading=18,
                             spaceBefore=10, spaceAfter=6, textColor=ACCENT),
        "h2": ParagraphStyle("h2", parent=body, fontName="Helvetica-Bold", fontSize=11, leading=14,
                             spaceBefore=8, spaceAfter=4),
        "body": body,
        "small": ParagraphStyle("small", parent=body, fontSize=8, leading=10.5, textColor=MUTED),
        "cell": ParagraphStyle("cell", parent=body, fontSize=7.8, leading=9.5, spaceAfter=0),
        "cellb": ParagraphStyle("cellb", parent=body, fontSize=7.8, leading=9.5, spaceAfter=0,
                                fontName="Helvetica-Bold"),
    }


def _md(text: str) -> str:
    """Escape text and convert **bold** / `code` markdown to ReportLab markup."""
    # Base-14 PDF fonts only cover WinAnsi; map the few symbols we use outside it.
    for src, dst in (("★", "(selected)"), ("Δ", "Diff."), ("↔", "vs"), ("×", "x"), ("→", "->")):
        text = text.replace(src, dst)
    out = escape(text)
    out = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", out)
    out = re.sub(r"`(.+?)`", r"<font face='Courier'>\1</font>", out)
    return out


def _bullets(items: list[str], st) -> ListFlowable:
    return ListFlowable([ListItem(Paragraph(_md(i), st["body"]), leftIndent=10) for i in items],
                        bulletType="bullet", start="•", leftIndent=10, bulletFontSize=8)


def _table(df: pd.DataFrame, st, col_widths=None, header_bg=ACCENT) -> Table:
    data = [[Paragraph(_md(str(c)), st["cellb"]) for c in df.columns]]
    data += [[Paragraph(_md(str(v)), st["cell"]) for v in row] for row in df.itertuples(index=False)]
    t = Table(data, colWidths=col_widths, repeatRows=1, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EAF2F8")),
        ("LINEBELOW", (0, 0), (-1, 0), 0.8, header_bg),
        ("LINEBELOW", (0, 1), (-1, -1), 0.3, RULE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]))
    return t


def _image(png: bytes, width: float) -> Image:
    img = Image(io.BytesIO(png))
    ratio = img.imageHeight / img.imageWidth
    img.drawWidth, img.drawHeight = width, width * ratio
    return img


def _footer(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(MARGIN, 10 * mm, "Titanic Survival Prediction — model report")
    canvas.drawRightString(PAGE_W - MARGIN, 10 * mm, f"Page {doc.page}")
    canvas.setStrokeColor(RULE)
    canvas.line(MARGIN, 13 * mm, PAGE_W - MARGIN, 13 * mm)
    canvas.restoreState()


def build_pdf(results: dict) -> bytes:
    """Build the full report and return the PDF bytes."""
    try:
        return _build(results)
    except Exception as exc:  # surface any ReportLab/Matplotlib failure uniformly
        log.exception("PDF generation failed")
        raise ReportGenerationError(f"Could not generate PDF report: {exc}") from exc


def _build(results: dict) -> bytes:
    st = _styles()
    sel_key, sel = selected(results)
    cfg, data = results["config"], results["data"]
    prof = data["profile"]
    story: list = []

    # 1. Title -------------------------------------------------------------
    story += [Paragraph("Titanic Survival Prediction", st["title"]),
              Paragraph(f"Machine learning model report · generated {datetime.now():%d %B %Y, %H:%M} · "
                        f"experiment run {escape(results['generated_at'])} UTC", st["subtitle"])]
    tile_val = ParagraphStyle("tile", parent=st["body"], fontName="Helvetica-Bold", fontSize=11,
                              leading=13, spaceAfter=0)
    values = [sel["display_name"], f"{sel['selection_mean']:.3f} ± {sel['selection_std']:.3f}",
              f"{sel['test_metrics']['roc_auc']:.3f}", f"{sel['test_metrics']['accuracy']:.1%}",
              f"{sel['test_metrics']['mcc']:.3f}"]
    tiles = [["Selected model", "CV ROC-AUC", "Test ROC-AUC", "Test accuracy", "Test MCC"],
             [Paragraph(escape(v), tile_val) for v in values]]
    t = Table(tiles, colWidths=[CONTENT_W * w for w in (0.28, 0.21, 0.17, 0.17, 0.17)], hAlign="LEFT")
    t.setStyle(TableStyle([
        ("FONT", (0, 0), (-1, 0), "Helvetica", 7.5), ("TEXTCOLOR", (0, 0), (-1, 0), MUTED),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BOX", (0, 0), (-1, -1), 0.5, RULE), ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
        ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
    story += [t, Spacer(1, 8)]

    # 2. Executive summary --------------------------------------------------
    story.append(Paragraph("1. Executive summary", st["h1"]))
    for sec in build_summary(results):
        block = [Paragraph(_md(sec.heading), st["h2"])]
        block += [Paragraph(_md(p), st["body"]) for p in sec.paragraphs]
        if sec.bullets:
            block.append(_bullets(sec.bullets, st))
        story.append(KeepTogether(block[:2]) if len(block) > 1 else block[0])
        story += block[2:] if len(block) > 2 else []

    # 3. Dataset ------------------------------------------------------------
    story += [PageBreak(), Paragraph("2. Dataset", st["h1"])]
    ov = prof["overview"]
    story.append(Paragraph(
        f"Source: <font face='Courier' size='8'>{escape(data['source'])}</font> "
        f"(SHA-256 {data['sha256'][:16]}…). {ov['n_rows']:,} rows × {ov['n_columns']} columns; "
        f"target <b>{config.TARGET}</b> with {ov['target_counts'].get('1', ov['target_counts'].get(1, 0))} "
        f"survivors ({ov['positive_rate']:.1%}).", st["body"]))
    feat = pd.DataFrame({"Feature": list(data["feature_descriptions"]),
                         "Description": list(data["feature_descriptions"].values())})
    story += [Paragraph("Model features", st["h2"]), _table(feat, st, [28 * mm, CONTENT_W - 28 * mm])]
    story += [Spacer(1, 6), _image(static.survival_rates_png(prof), CONTENT_W),
              Paragraph("Bars show observed survival rates; whiskers are Wilson 95% intervals.", st["small"])]

    # 4. Data quality -------------------------------------------------------
    story.append(Paragraph("3. Data quality findings", st["h1"]))
    miss = pd.DataFrame(prof["missing"])
    miss = miss[miss["missing"] > 0][["column", "missing", "percent"]]
    miss.columns = ["Column", "Missing", "%"]
    story += [Paragraph("Missing values", st["h2"]), _table(miss, st, [40 * mm, 25 * mm, 20 * mm])]
    red = pd.DataFrame(prof["redundancy"])
    red["holds"] = red["holds"].map({True: "verified", False: "NOT exact", None: "—"})
    red = red[["column", "relation", "holds", "reason"]]
    red.columns = ["Excluded column", "Relationship tested", "Result", "Reason"]
    story += [Paragraph("Excluded columns (redundancy and leakage checks)", st["h2"]),
              _table(red, st, [24 * mm, 52 * mm, 17 * mm, CONTENT_W - 93 * mm])]
    out = pd.DataFrame(prof["outliers"])[["column", "upper_fence", "n_outliers", "percent", "max"]]
    out = out.round(2)
    out.columns = ["Column", "Upper Tukey fence", "Outliers", "%", "Max"]
    story += [Paragraph("Outliers (1.5 × IQR rule)", st["h2"]), _table(out, st),
              Paragraph(f"{ov['n_duplicate_rows']} exact duplicate rows were retained: without names or "
                        "tickets they are indistinguishable from distinct passengers with identical "
                        "attributes. Fare outliers are genuine luxury tickets; they are kept and "
                        "log-transformed rather than removed.", st["body"])]

    # 5. Preprocessing ------------------------------------------------------
    story.append(Paragraph("4. Preprocessing methodology", st["h1"]))
    story.append(_bullets([
        "Row-wise feature engineering (log fare, child flag) with no learned statistics.",
        "Numeric: median imputation with missing indicator, then standardisation.",
        "Categorical (pclass, sex, embarked): most-frequent imputation, one-hot encoding "
        "(unknown categories ignored at prediction time).",
        "All steps live in one scikit-learn Pipeline fitted only on training folds — the test set "
        "never influences imputation values, scaling or encoder categories.",
    ], st))

    # 6. Training config ----------------------------------------------------
    story.append(Paragraph("5. Training and model-selection configuration", st["h1"]))
    conf = pd.DataFrame({"Setting": ["Random seed", "Train / test split", "Inner CV", "Search",
                                     "Selection metric", "Selection estimate", "Bootstrap",
                                     "Decision threshold"],
                         "Value": [cfg["random_state"],
                                   f"{data['n_train']} / {data['n_test']} (stratified, test {cfg['test_size']:.0%})",
                                   f"{cfg['cv_folds']}-fold stratified, shuffled",
                                   f"RandomizedSearchCV, up to {cfg['n_iter']} candidates per model",
                                   METRIC_LABELS[cfg["selection_metric"]],
                                   results["selection"]["estimate"],
                                   f"{cfg['n_bootstrap']:,} stratified resamples of the test set",
                                   cfg["decision_threshold"]]})
    story.append(_table(conf, st, [40 * mm, CONTENT_W - 40 * mm]))

    # 7. Comparison ---------------------------------------------------------
    story += [PageBreak(), Paragraph("6. Model comparison", st["h1"])]
    comp = comparison_table(results)[["Model", "CV ROC-AUC (selection)", "Test ROC-AUC [95% CI]",
                                      "Test PR-AUC", "Accuracy", "Balanced acc.", "F1", "MCC"]]
    story.append(_table(comp, st, [32 * mm, 26 * mm, 33 * mm, 17 * mm, 17 * mm, 19 * mm, 14 * mm,
                                   CONTENT_W - 158 * mm]))
    story += [Spacer(1, 6), _image(static.metric_comparison_png(results), CONTENT_W)]
    params = pd.DataFrame([{"Model": m["display_name"],
                            "Tuned hyperparameters": ", ".join(f"{k}={_fmt(v)}" for k, v in m["best_params"].items()),
                            "Candidates": m["n_candidates"]} for m in results["models"].values()])
    story += [Paragraph("Tuned hyperparameters", st["h2"]),
              _table(params, st, [34 * mm, CONTENT_W - 54 * mm, 20 * mm])]

    # 8. Selection ----------------------------------------------------------
    story.append(Paragraph("7. Selected model and rationale", st["h1"]))
    story.append(Paragraph(_md(
        f"**{sel['display_name']}** was selected because it achieved the highest "
        f"{METRIC_LABELS[cfg['selection_metric']]} under {results['selection']['estimate']} "
        f"({sel['selection_mean']:.3f} ± {sel['selection_std']:.3f}). The choice was made before any "
        "test-set evaluation. " + config.SELECTION_METRIC_RATIONALE), st["body"]))
    diffs = results["selected_diagnostics"]["paired_auc_vs_others"]
    if diffs:
        d = pd.DataFrame([{"Compared with": results["models"][k]["display_name"],
                           "Δ test ROC-AUC": f"{v['difference']:+.3f}",
                           "Paired bootstrap 95% CI": f"[{v['low']:+.3f}, {v['high']:+.3f}]",
                           "Share of resamples where selected is better": f"{v['share_a_better']:.0%}"}
                          for k, v in diffs.items()])
        story += [Paragraph("Is the selected model distinguishable on the test set?", st["h2"]), _table(d, st)]

    # 9. Evaluation ---------------------------------------------------------
    story += [PageBreak(), Paragraph("8. Evaluation metrics (selected model, test set)", st["h1"])]
    ci = sel["test_ci"]
    ev = pd.DataFrame([{"Metric": METRIC_LABELS[k], "Value": f"{ci[k]['estimate']:.3f}",
                        "95% CI": f"{ci[k]['low']:.3f} – {ci[k]['high']:.3f}",
                        "Meaning": METRIC_EXPLANATIONS[k]} for k in ci])
    story.append(_table(ev, st, [30 * mm, 15 * mm, 25 * mm, CONTENT_W - 70 * mm]))
    story.append(Paragraph("Confidence intervals: percentile bootstrap over test rows, resampled within "
                           "each class. They reflect test-sample uncertainty only, not the variability "
                           "of re-training on new data.", st["small"]))

    # 10. Confusion + 11. curves --------------------------------------------
    cm = sel["confusion"]
    story.append(KeepTogether([
        Paragraph("9. Confusion matrix", st["h1"]),
        _image(static.confusion_png(cm, sel["display_name"]), 80 * mm),
        Paragraph(f"Of {cm['tp'] + cm['fn']} survivors in the test set, {cm['tp']} were identified "
                  f"and {cm['fn']} missed; of {cm['tn'] + cm['fp']} non-survivors, {cm['fp']} were "
                  "wrongly predicted to survive.", st["body"])]))
    story.append(KeepTogether([
        Paragraph("10. ROC and precision–recall curves", st["h1"]),
        _image(static.roc_pr_png(results), CONTENT_W),
        Paragraph(f"Dotted lines mark chance: AUC 0.5 and precision equal to the test survival "
                  f"rate ({sel['curves']['prevalence']:.2f}).", st["small"])]))

    # 12. Importance ----------------------------------------------------------
    story += [PageBreak(), Paragraph("11. Feature importance and interpretation", st["h1"])]
    pi = sel["permutation_importance"]
    story += [_image(static.importance_png(pi["values"], "mean",
                                           f"Permutation importance — {sel['display_name']}", "std"),
                     CONTENT_W * 0.85),
              Paragraph(f"Mean drop in test ROC-AUC when each raw input is shuffled ({pi['n_repeats']} "
                        "repeats); whiskers show ±1 SD.", st["small"])]
    imp = sel["importance"]
    if imp and imp["kind"] == "impurity":
        story.append(_image(static.importance_png(imp["values"], "importance",
                                                  "Impurity-based importance (training data)"), CONTENT_W * 0.85))
    lr = results["models"].get("logistic_regression")
    if lr and lr["importance"]:
        story.append(_image(static.importance_png(lr["importance"]["values"], "coefficient",
                                                  "Logistic Regression coefficients (log-odds, standardised inputs)"),
                            CONTENT_W * 0.85))
    story.append(_bullets([
        "Permutation importance measures how much held-out performance drops when one raw input is "
        "shuffled; it applies to any model and groups one-hot dummies of the same column.",
        "Impurity importance (trees) is computed on training data and favours continuous, "
        "high-cardinality inputs such as fare and age.",
        "Logistic coefficients are signed log-odds effects per standard deviation (or per category vs. "
        "the others); blue increases and vermillion decreases predicted survival.",
        "Correlated inputs (class ↔ fare, sex ↔ age-group) split credit, so no single ranking is "
        "definitive. Importance describes model behaviour, not causation.",
    ], st))

    # 13/14. Findings & limitations ------------------------------------------
    sections = {s.heading: s for s in build_summary(results)}
    for num, head in (("12", "Main findings"), ("13", "Limitations"), ("14", "Recommendations and future work")):
        s = sections.get(head)
        if s:
            story += [Paragraph(f"{num}. {head}", st["h1"]), _bullets(s.bullets, st)]
    env = results["environment"]
    story += [Spacer(1, 8), Paragraph(
        "Environment: " + ", ".join(f"{k} {escape(str(v))}" for k, v in env.items()) +
        f". Total experiment runtime {results['runtime_seconds']:.0f} s.", st["small"])]

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=MARGIN, rightMargin=MARGIN, topMargin=MARGIN,
                            bottomMargin=20 * mm, title="Titanic Survival Prediction — Model Report",
                            author="Titanic ML project")
    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
    return buf.getvalue()


def _fmt(v) -> str:
    return f"{v:.4g}" if isinstance(v, float) else str(v)


def write_pdf(results: dict, path: Path = config.PDF_REPORT_FILE) -> Path:
    """Generate the PDF and write it to ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(build_pdf(results))
    log.info("PDF written", extra={"fields": {"path": path}})
    return path


if __name__ == "__main__":
    from src.utils import read_json
    write_pdf(read_json(config.RESULTS_FILE))
