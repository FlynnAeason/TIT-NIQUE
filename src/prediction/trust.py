"""Trust aids for a single prediction.

* :func:`model_vote` asks every trained model, not just the selected one. When
  models built in different ways agree, the answer is less likely to be an
  artefact of one model's quirks; a split vote flags a genuinely uncertain case.
* :func:`similar_passengers` looks up real passengers in the data who resemble
  the one described, so the model's number can be compared with what actually
  happened to people like them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

import config
from src.data.profiling import wilson_interval
from src.prediction.predict import LoadedModel, predict_one


def model_vote(models: dict[str, LoadedModel], passenger: dict[str, Any]) -> dict[str, Any]:
    """Every model's chance and answer, plus a plain summary of how much they agree."""
    votes = []
    for key, m in models.items():
        out = predict_one(m, passenger)
        votes.append({"key": key, "model": m.display_name, "prob": out["prob_survived"],
                      "survived": out["prediction"] == config.POSITIVE_LABEL})
    n, yes = len(votes), sum(v["survived"] for v in votes)
    majority = max(yes, n - yes)
    if n == 0:
        level = "none"
    elif majority == n:
        level = "unanimous"
    elif majority / n >= 0.75:
        level = "strong"
    else:
        level = "split"
    probs = [v["prob"] for v in votes]
    return {"votes": votes, "n": n, "n_survived": yes, "level": level,
            "low": min(probs) if probs else float("nan"), "high": max(probs) if probs else float("nan")}


@dataclass
class SimilarResult:
    matches: pd.DataFrame
    criteria: list[str] = field(default_factory=list)
    n: int = 0
    survived: int = 0
    rate: float = float("nan")
    ci: tuple[float, float] = (float("nan"), float("nan"))


def similar_passengers(df: pd.DataFrame, passenger: dict[str, Any], min_n: int = 10) -> SimilarResult:
    """Real passengers resembling ``passenger``, loosening the match until at least ``min_n`` are found.

    Matching always uses sex and ticket class (the two strongest signals). It then
    tries, from most to least specific: age within 5 years and the same "alone vs.
    with family" status; age within 5 years; age within 10 years; sex and class only.
    When the age is unknown, only passengers whose age is also unknown are used for
    the age-based steps.
    """
    base = df[(df["sex"] == passenger["sex"]) & (df["pclass"] == int(passenger["pclass"]))]
    alone = (int(passenger["sibsp"]) + int(passenger["parch"])) == 0
    family_mask = ((base["sibsp"] + base["parch"]) == 0) == alone
    family_txt = "travelling alone" if alone else "travelling with family"
    sex_txt = "women" if passenger["sex"] == "female" else "men"
    class_txt = f"in {['1st', '2nd', '3rd'][int(passenger['pclass']) - 1]} class"
    age = passenger.get("age")

    steps: list[tuple[pd.Series, list[str]]] = []
    if age is None or pd.isna(age):
        unknown = base["age"].isna()
        steps += [(unknown & family_mask, [sex_txt, class_txt, "age not recorded", family_txt]),
                  (unknown, [sex_txt, class_txt, "age not recorded"])]
    else:
        a = float(age)
        for width, with_family in ((5, True), (5, False), (10, False)):
            near = base["age"].between(a - width, a + width)
            crit = [sex_txt, class_txt, f"aged {max(0, a - width):.0f}–{a + width:.0f}"]
            if with_family:
                steps.append((near & family_mask, crit + [family_txt]))
            else:
                steps.append((near, crit))
    steps.append((pd.Series(True, index=base.index), [sex_txt, class_txt]))

    for mask, criteria in steps:
        matches = base[mask]
        if len(matches) >= min_n or criteria == [sex_txt, class_txt]:
            break
    n, k = len(matches), int(matches[config.TARGET].sum())
    return SimilarResult(matches=matches, criteria=criteria, n=n, survived=k,
                         rate=k / n if n else float("nan"), ci=wilson_interval(k, n))
