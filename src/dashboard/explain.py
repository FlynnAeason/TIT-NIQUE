"""Plain-language wording for the dashboard's Simple mode.

Every sentence is built from real numbers passed in; nothing here contains
results. Technical terms appear in brackets after the plain wording so readers
can connect the two.
"""

from __future__ import annotations

from math import gcd

import streamlit as st

# Denominators people read easily ("4 out of 5", "9 out of 10").
_NICE = (2, 3, 4, 5, 10)


def in_words(x: float) -> str:
    """Express a proportion as a simple fraction: 0.799 -> 'about 4 out of 5'."""
    if x != x:  # NaN
        return "an unknown share"
    if x >= 0.97:
        return "almost all"
    if x <= 0.03:
        return "almost none"
    for d in _NICE:
        n = round(x * d)
        if 0 < n < d and abs(n / d - x) <= 0.035:
            g = gcd(n, d)
            return f"about {n // g} out of {d // g}"
    return f"about {round(x * 10)} out of 10"


PLAIN_FEATURES: dict[str, str] = {
    "sex": "Sex",
    "pclass": "Ticket class",
    "age": "Age",
    "fare": "Fare paid",
    "sibsp": "Siblings / spouses aboard",
    "parch": "Parents / children aboard",
    "embarked": "Port of embarkation",
}


def plain_feature(name: str) -> str:
    return PLAIN_FEATURES.get(name, name)


# Plain label, technical name.
PLAIN_LABELS: dict[str, tuple[str, str]] = {
    "accuracy": ("How often it's right", "accuracy"),
    "recall": ("Survivors it finds", "recall"),
    "specificity": ("Non-survivors it rules out", "specificity"),
    "precision": ("When it says 'survived', how often it's true", "precision"),
    "roc_auc": ("Ranking skill", "ROC-AUC"),
}


def metric_sentence(metric: str, value: float, with_value: bool = True) -> str:
    """One plain sentence explaining a metric value.

    ``with_value=False`` drops the bracketed number when it is already shown nearby.
    """
    w = in_words(value)
    v = f" ({value:.0%})" if with_value else ""
    return {
        "accuracy": f"Right {w} times{v}.",
        "recall": f"Finds {w} of the people who actually survived{v}.",
        "specificity": f"Correctly rules out {w} of the people who did not survive{v}.",
        "precision": f"When it predicts 'survived', it is right {w} times{v}.",
        "roc_auc": (f"Shown one survivor and one non-survivor, it gives the survivor the higher "
                    f"chance {w} times ({value:.2f}; 0.5 would be a coin flip)."),
    }[metric]


def ladder_summary(rows: list[dict]) -> tuple[str, str]:
    """(short sentence, longer explanation) comparing the model with the best simple rule."""
    model = next(r for r in rows if r["is_model"])
    rules = [r for r in rows if not r["is_model"]]
    best = max(rules[1:] or rules, key=lambda r: r["accuracy"])  # skip "everyone dies" when possible
    gain = model["correct"] - best["correct"]
    short = (f"The simple rule *'{best['step'].lower()}'* already gets {best['accuracy']:.0%} right; "
             f"the model gets {model['accuracy']:.0%}")
    if gain <= 0:
        short += " — no more passengers right than the rule when giving yes/no answers."
    elif gain <= 0.03 * model["n"]:
        short += f" — only {gain} more passenger{'s' if gain != 1 else ''} out of {model['n']}."
    else:
        short += f" — {gain} more passengers out of {model['n']}."
    if gain <= 0.03 * model["n"]:
        long = ("Most of what the model knows is captured by that one rule. What the model adds is a "
                "**graded chance for each person** — it can tell 'probably survives' from 'almost "
                "certainly survives', and rank who is most at risk within a group — which a yes/no rule "
                "cannot do.")
    else:
        long = ("The model picks up patterns beyond any single rule, such as how age, family size and "
                "class combine.")
    return short, long


def grade(auc: float) -> str:
    """Rough verbal grade for a ROC-AUC value (conventional bands)."""
    if auc >= 0.9:
        return "excellent"
    if auc >= 0.8:
        return "good"
    if auc >= 0.7:
        return "fair"
    return "weak"


GLOSSARY: dict[str, str] = {
    "Model": "A set of rules learned from past examples, used to estimate the chance of survival.",
    "Training / test data": "The model learns from 80% of the passengers (training) and is graded "
                            "on the other 20% it has never seen (test).",
    "Probability": "The model's estimated chance of survival, from 0% to 100%.",
    "Threshold": "The cut-off that turns a probability into a yes/no answer. Here: 50%.",
    "Accuracy": "Share of passengers the model gets right.",
    "Recall": "Of the people who survived, the share the model found.",
    "Specificity": "Of the people who died, the share the model correctly ruled out.",
    "Precision": "Of the people the model said would survive, the share who did.",
    "ROC-AUC": "How well the model ranks survivors above non-survivors. 0.5 = coin flip, 1 = perfect.",
    "Confusion matrix": "A 2×2 table counting right and wrong answers of each kind.",
    "Cross-validation": "Grading a model several times on different slices of the training data, "
                        "for a fairer score than a single try.",
    "Confidence interval": "A range that probably contains the true score; wide ranges mean "
                           "less certainty.",
    "Feature": "One piece of passenger information the model uses, such as age or ticket class.",
    "Feature importance": "How much the model's accuracy drops when one feature is scrambled. "
                          "It shows what the model relies on, not what caused survival.",
    "Overfitting": "When a model memorises its training data and does worse on new data.",
    "Data leakage": "When the answer sneaks into the inputs, making a model look better than it is.",
}


def detailed() -> bool:
    """True when the reader switched on Detailed mode in the sidebar."""
    return bool(st.session_state.get("detailed", False))


def detail_hint() -> None:
    """Small pointer telling Simple-mode readers where the rest is."""
    if not detailed():
        st.caption(":material/tune: Switch on **Detailed mode** in the sidebar for the full "
                   "technical numbers on this page.")
