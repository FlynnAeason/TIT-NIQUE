"""What it relies on: which passenger details drive the model."""

import pandas as pd
import streamlit as st

from src.dashboard import state
from src.dashboard.explain import detail_hint, detailed, plain_feature
from src.visualization import charts

st.title("What it relies on")
st.markdown("To see which details matter, we **scramble one detail at a time** (for example, shuffle "
            "everyone's age) and check how much worse the model gets. The bigger the drop, the more "
            "the model depends on that detail.")

res = state.require_results()
names = state.model_names(res)
sel_key = res["selection"]["selected_model"]

if detailed():
    key = st.segmented_control("Model", list(names),
                               format_func=lambda k: names[k] + (" ★" if k == sel_key else ""),
                               default=sel_key, required=True, key="xp_model")
else:
    key = sel_key
m = res["models"][key]
pi = m["permutation_importance"]

plain_values = dict(pi["values"])
plain_values["feature"] = [plain_feature(f) for f in pi["values"]["feature"]]
st.plotly_chart(charts.importance_bar(plain_values, "mean", "How much worse the model gets when a detail is scrambled",
                                      error_col="std" if detailed() else None,
                                      x_title="Drop in ranking skill (ROC-AUC)"),
                alt="Permutation importance by passenger detail")

vals = pd.DataFrame(pi["values"])
top = vals.iloc[0]
useful = vals[vals["mean"] >= 0.01]["feature"].map(plain_feature).tolist()
minor = vals[vals["mean"] < 0.01]["feature"].map(plain_feature).tolist()
st.markdown(f"- **{plain_feature(top['feature'])}** matters most by a wide margin."
            if len(vals) > 1 and top["mean"] > 2 * vals.iloc[1]["mean"] else
            f"- **{plain_feature(top['feature'])}** matters most.")
if len(useful) > 1:
    st.markdown(f"- Also useful: {', '.join(useful[1:])}.")
if minor:
    st.markdown(f"- Adds little once the others are known: {', '.join(minor)}. That does not mean these "
                "details were irrelevant in 1912 — often they overlap with something the model already "
                "uses (for example, fare is closely linked to ticket class).")

st.warning("This shows what the **model** relies on, not what **caused** survival. A detail can matter "
           "to the model simply because it goes along with the real cause.",
           title="Pattern, not cause", icon=":material/psychology_alt:")

if not detailed():
    detail_hint()
    st.stop()

# ======================= Detailed mode =============================================
st.divider()
st.subheader("Model-specific views")
st.caption(f"Each raw column is shuffled {pi['n_repeats']} times on held-out data (bars: mean drop in ROC-AUC, "
           "whiskers: ±1 SD). Shuffling `age` also scrambles the derived child flag and missing-age "
           "indicator, so each bar covers a whole input. Values near zero or negative mean the model "
           "does not need that column given the others.")
imp = m["importance"]
if imp is None:
    st.info(f"{names[key]} has no built-in importance measure; rely on permutation importance above.",
            icon=":material/info:")
elif imp["kind"] == "impurity":
    st.plotly_chart(charts.importance_bar(imp["values"], "importance", "Impurity-based importance (training data)",
                                          x_title="Mean decrease in impurity"),
                    alt="Impurity-based feature importance")
    st.caption("Computed on training data over the encoded features. It is biased toward continuous, "
               "many-valued inputs (fare, age) and can credit noise the model over-fitted.")
else:
    st.plotly_chart(charts.importance_bar(imp["values"], "coefficient", "Logistic Regression coefficients",
                                          x_title="Log-odds change (standardised inputs)"),
                    alt="Logistic regression coefficients")
    st.caption("Blue raises and red-orange lowers the predicted odds of survival. Numeric inputs are "
               "standardised, so a coefficient is the log-odds change per 1 SD. One-hot categories "
               "share a redundant baseline, so compare categories within a feature, not alone.")
    st.dataframe(pd.DataFrame(imp["values"]).round(4), hide_index=True, alt="Coefficients and odds ratios",
                 column_config={"odds_ratio": st.column_config.NumberColumn("odds ratio", format="%.3f")})
    st.caption(f"Intercept: {imp['intercept']:.3f}. Odds ratio = exp(coefficient).")

st.subheader("Reading these charts responsibly")
st.markdown("""
- **Different questions.** Permutation importance asks *how much does held-out performance depend on this
  input?* Impurity importance asks *how often did trees split on it during training?* Coefficients ask
  *how does the log-odds move per unit of this input, holding the others fixed?*
- **Shared information.** `pclass`, `fare` and port are associated, as are `sex` and the child flag. When
  inputs overlap, a model can lean on either, so importance is split or shifted between them. This is why
  the redundant columns (`who`, `adult_male`, `class`, `alone`) were removed before modelling.
- **Not causal.** These scores describe the model's behaviour on this dataset only.
""")
