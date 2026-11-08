"""Exploratory data analysis computed from the actual data.

Every statistic or redundancy claim shown in the dashboard and the PDF report is
produced here, so nothing about the dataset is hard-coded elsewhere.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

import config


def dataset_overview(df: pd.DataFrame) -> dict[str, Any]:
    """Return dimensions, dtypes, target balance and duplicate counts."""
    target_counts = df[config.TARGET].value_counts().sort_index()
    return {
        "n_rows": int(len(df)),
        "n_columns": int(df.shape[1]),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "target_counts": {int(k): int(v) for k, v in target_counts.items()},
        "positive_rate": float(df[config.TARGET].mean()),
        "n_duplicate_rows": int(df.duplicated().sum()),
        "n_duplicate_model_rows": int(df[config.MODEL_FEATURES + [config.TARGET]].duplicated().sum()),
    }


def missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Missing-value count and percentage per column, sorted descending."""
    counts = df.isna().sum()
    return (pd.DataFrame({"column": counts.index, "missing": counts.values,
                          "percent": (counts.values / len(df) * 100).round(2)})
            .sort_values("missing", ascending=False).reset_index(drop=True))


def numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Descriptive statistics plus skewness for numeric columns."""
    num = df.select_dtypes("number")
    summary = num.describe().T
    summary["skew"] = num.skew()
    return summary.round(3)


def outlier_summary(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """Count observations outside Tukey fences (1.5 * IQR) per numeric column."""
    columns = columns or ["age", "fare", "sibsp", "parch"]
    rows = []
    for col in columns:
        s = df[col].dropna()
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_out = int(((s < lo) | (s > hi)).sum())
        rows.append({"column": col, "q1": q1, "q3": q3, "lower_fence": lo, "upper_fence": hi,
                     "n_outliers": n_out, "percent": round(100 * n_out / len(s), 2),
                     "max": s.max()})
    return pd.DataFrame(rows)


def survival_rate_by(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Survival rate, count and Wilson 95% interval per level of ``column``."""
    keys = df[column].astype("object").where(df[column].notna(), "missing").map(str)
    grouped = df.groupby(keys.rename(column), observed=True)[config.TARGET].agg(["mean", "count", "sum"])
    grouped = grouped.reset_index().rename(columns={"mean": "survival_rate", "sum": "survivors"})
    lo, hi = zip(*(wilson_interval(int(s), int(n)) for s, n in zip(grouped["survivors"], grouped["count"])))
    grouped["ci_low"], grouped["ci_high"] = lo, hi
    return grouped


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (float(centre - half), float(centre + half))


def cramers_v(x: pd.Series, y: pd.Series) -> float:
    """Bias-corrected Cramer's V between two categorical series."""
    table = pd.crosstab(x, y)
    if table.shape[0] < 2 or table.shape[1] < 2:
        return float("nan")
    chi2 = stats.chi2_contingency(table, correction=False)[0]
    n = table.to_numpy().sum()
    phi2 = chi2 / n
    r, k = table.shape
    phi2corr = max(0.0, phi2 - (k - 1) * (r - 1) / (n - 1))
    rcorr = r - (r - 1) ** 2 / (n - 1)
    kcorr = k - (k - 1) ** 2 / (n - 1)
    denom = min(kcorr - 1, rcorr - 1)
    return float(np.sqrt(phi2corr / denom)) if denom > 0 else float("nan")


def association_matrix(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """Mixed-type association matrix.

    Numeric-numeric pairs use |Spearman rho|; any pair involving a categorical
    column uses bias-corrected Cramer's V (numeric columns are binned into
    quintiles for that purpose). All values lie in [0, 1], which makes one
    heatmap readable even though the measures differ; the dashboard states this.
    """
    columns = columns or ["survived", "pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]
    numeric = {"age", "fare", "sibsp", "parch"}
    binned = {}
    for c in columns:
        if c in numeric:
            binned[c] = pd.qcut(df[c], q=5, duplicates="drop").astype(str).where(df[c].notna())
        else:
            binned[c] = df[c].astype(str).where(df[c].notna())
    mat = pd.DataFrame(np.eye(len(columns)), index=columns, columns=columns)
    for i, a in enumerate(columns):
        for b in columns[i + 1:]:
            if a in numeric and b in numeric:
                val = abs(df[[a, b]].corr(method="spearman").iloc[0, 1])
            else:
                mask = binned[a].notna() & binned[b].notna()
                val = cramers_v(binned[a][mask], binned[b][mask])
            mat.loc[a, b] = mat.loc[b, a] = val
    return mat.round(3)


def numeric_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """Spearman correlation among numeric columns and the target."""
    cols = ["survived", "pclass", "age", "sibsp", "parch", "fare"]
    return df[cols].corr(method="spearman").round(3)


def redundancy_checks(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Verify, from the data itself, the redundancy/leakage claims in config.

    Each entry reports whether the stated relationship holds exactly, so the UI
    and report never assert something the data does not support.
    """
    checks: list[dict[str, Any]] = []

    def add(column: str, relation: str, holds: bool | None, detail: str) -> None:
        checks.append({"column": column, "relation": relation, "holds": holds,
                       "detail": detail, "reason": config.EXCLUDED_FEATURES.get(column, "")})

    if {"alive", "survived"} <= set(df.columns):
        holds = bool((df["alive"].map({"yes": 1, "no": 0}) == df["survived"]).all())
        add("alive", "alive == ('yes' if survived else 'no')", holds, "Identical to the target.")
    if {"class", "pclass"} <= set(df.columns):
        mapping = {"First": 1, "Second": 2, "Third": 3}
        holds = bool((df["class"].map(mapping) == df["pclass"]).all())
        add("class", "class <-> pclass one-to-one", holds, "Same information, different encoding.")
    if {"who", "sex", "age"} <= set(df.columns):
        derived = np.where(df["age"] < config.CHILD_AGE_THRESHOLD, "child",
                           np.where(df["sex"] == "male", "man", "woman"))
        holds = bool((derived == df["who"]).all())
        add("who", f"who == 'child' if age < {config.CHILD_AGE_THRESHOLD:g} else man/woman by sex",
            holds, "Missing ages are labelled man/woman purely from sex.")
    if {"adult_male", "who"} <= set(df.columns):
        holds = bool(((df["who"] == "man") == df["adult_male"]).all())
        add("adult_male", "adult_male == (who == 'man')", holds, "Re-encodes `who`.")
    if {"alone", "sibsp", "parch"} <= set(df.columns):
        holds = bool((((df["sibsp"] + df["parch"]) == 0) == df["alone"]).all())
        add("alone", "alone == (sibsp + parch == 0)", holds, "Derived from family counts.")
    if {"embark_town", "embarked"} <= set(df.columns):
        mapped = df["embarked"].map(config.EMBARKED_NAMES)
        same = (mapped == df["embark_town"]) | (mapped.isna() & df["embark_town"].isna())
        add("embark_town", "embark_town <-> embarked one-to-one", bool(same.all()),
            "Port code vs port name.")
    if "deck" in df.columns:
        pct = float(df["deck"].isna().mean() * 100)
        add("deck", "missing share", None, f"{pct:.1f}% missing.")
    return checks


def full_profile(df: pd.DataFrame) -> dict[str, Any]:
    """Bundle the profile used by training artifacts and the PDF report."""
    return {
        "overview": dataset_overview(df),
        "missing": missing_values(df).to_dict(orient="records"),
        "outliers": outlier_summary(df).to_dict(orient="records"),
        "redundancy": redundancy_checks(df),
        "survival_by": {c: survival_rate_by(df, c).to_dict(orient="records")
                        for c in ["sex", "pclass", "embarked", "who"] if c in df.columns},
    }
