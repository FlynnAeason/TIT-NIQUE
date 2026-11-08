"""The data: who was on board and which groups survived."""

import pandas as pd
import streamlit as st

import config
from src.data import profiling
from src.dashboard import state
from src.dashboard.explain import detail_hint, detailed, in_words
from src.visualization import charts

st.title("Who was on board")
st.caption("891 real passengers. Use the filters to focus on a group — every number and chart on "
           "this page updates. (The model is always trained on the same fixed data, whatever you pick.)")

df = state.require_data()

with st.container(border=True):
    c1, c2, c3, c4 = st.columns([0.8, 1, 1.8, 1.6], vertical_alignment="bottom")
    pclass = c1.pills("Ticket class", [1, 2, 3], selection_mode="multi", default=[1, 2, 3], key="ex_pclass")
    sex = c2.pills("Sex", ["female", "male"], selection_mode="multi", default=["female", "male"], key="ex_sex")
    embarked = c3.pills("Port", ["C", "Q", "S"], selection_mode="multi", default=["C", "Q", "S"],
                        format_func=lambda p: config.EMBARKED_NAMES[p], key="ex_port")
    age = c4.slider("Age range", 0, 80, (0, 80), key="ex_age")
    keep_missing_age = c4.toggle("Include passengers with unknown age", value=True, key="ex_age_na")

mask = df["pclass"].isin(pclass or []) & df["sex"].isin(sex or [])
mask &= df["embarked"].isin(embarked or []) | df["embarked"].isna()
mask &= df["age"].between(*age) | (df["age"].isna() & keep_missing_age)
fdf = df[mask]

if fdf.empty:
    st.warning("No passengers match these filters. Try selecting more options.", icon=":material/filter_alt_off:")
    st.stop()

rate = fdf[config.TARGET].mean()
with st.container(horizontal=True):
    st.metric("Passengers shown", f"{len(fdf):,}", border=True)
    st.metric("Survived", f"{rate:.0%}", border=True, help=f"{in_words(rate)} of the passengers shown.")
    st.metric("Typical age", f"{fdf['age'].median():.0f}" if fdf["age"].notna().any() else "—", border=True,
              help="Median: half the passengers were younger, half older.")
    st.metric("Typical fare", f"£{fdf['fare'].median():.2f}", border=True,
              help="Median ticket price in 1912 British pounds.")
if len(fdf) < len(df):
    st.caption(f"In this group, {in_words(rate)} survived — compared with "
               f"{df[config.TARGET].mean():.0%} of all passengers.")

tab_names = ["Who survived", "Browse the data"]
if detailed():
    tab_names += ["Distributions", "Data quality", "Associations"]
tabs = st.tabs(tab_names)

# ---- Who survived ----------------------------------------------------------
with tabs[0]:
    st.markdown("Each bar is the share of a group that survived. The thin line on top shows how "
                "sure we can be: **a long line means few people in that group**, so the bar could "
                "easily be higher or lower.")
    groups = fdf.assign(age_group=pd.cut(fdf["age"], [0, 15.99, 29.99, 44.99, 59.99, 200],
                                         labels=["0–15", "16–29", "30–44", "45–59", "60+"]))
    options = {"sex": "Sex", "pclass": "Ticket class", "age_group": "Age group", "embarked": "Port"}
    if detailed():
        options |= {"who": "Who (man/woman/child)", "sibsp": "Siblings/spouses", "parch": "Parents/children",
                    "deck": "Deck"}
    cols = st.columns(2)
    for i, (col, label) in enumerate(options.items()):
        with cols[i % 2]:
            st.plotly_chart(charts.survival_rate_bar(profiling.survival_rate_by(groups, col), col, label),
                            alt=f"Survival rate by {label}")
    st.caption("Ports: C = Cherbourg, Q = Queenstown, S = Southampton. 'missing' = not recorded."
               + (" `who` and `deck` are shown for exploration but are not used by the model." if detailed() else ""))

# ---- Browse ------------------------------------------------------------------
with tabs[1]:
    st.caption("One row per passenger. `survived` is ticked for survivors; `None` means the value "
               "was not recorded.")
    st.dataframe(fdf, hide_index=True, height=420, alt="Filtered Titanic passenger records",
                 column_config={"survived": st.column_config.CheckboxColumn("survived"),
                                "fare": st.column_config.NumberColumn("fare", format="%.2f")})
    st.download_button("Download these rows as CSV", fdf.to_csv(index=False), "titanic_filtered.csv",
                       "text/csv", icon=":material/download:")
    if detailed():
        with st.expander("Statistical summary"):
            st.dataframe(profiling.numeric_summary(fdf), alt="Descriptive statistics for numeric columns")

# ---- Detailed-only tabs ---------------------------------------------------------
if detailed():
    with tabs[2]:
        feature = st.segmented_control("Feature", ["age", "fare", "sibsp", "parch"], default="age",
                                       required=True, key="ex_feature")
        log_scale = feature == "fare" and st.toggle("Log-transform fare", value=True, key="ex_log")
        a, b = st.columns(2)
        with a:
            st.plotly_chart(charts.numeric_distribution(fdf, feature, log_x=log_scale,
                                                        bins=20 if feature in ("sibsp", "parch") else 30),
                            alt=f"Histogram of {feature} split by survival")
        with b:
            st.plotly_chart(charts.box_by_target(fdf, feature), alt=f"Box plot of {feature} by survival")
        st.caption("Left: how many passengers had each value, survivors and non-survivors overlaid. "
                   "Right: the box holds the middle half of passengers; dots are unusual values.")
        st.dataframe(profiling.outlier_summary(fdf).round(2), hide_index=True,
                     alt="Outlier counts by the 1.5 IQR rule")
        st.caption("Outliers use Tukey's 1.5 × IQR fences. High fares are genuine first-class tickets, "
                   "so they are kept; the model uses log(1 + fare) to reduce their leverage.")

    with tabs[3]:
        q1, q2 = st.columns([2, 3])
        with q1:
            st.plotly_chart(charts.missing_values_bar(profiling.missing_values(df)),
                            alt="Missing values per column")
            ov = profiling.dataset_overview(df)
            st.markdown(f"**Exact duplicate rows:** {ov['n_duplicate_rows']} "
                        f"({ov['n_duplicate_model_rows']} duplicates on model features + target). Without "
                        "names or ticket numbers these cannot be told apart from different passengers with "
                        "identical attributes, so they are kept rather than dropped.")
        with q2:
            st.markdown("**Leakage and redundancy checks** (verified on the data at runtime)")
            checks = pd.DataFrame(profiling.redundancy_checks(df))
            checks["holds"] = checks["holds"].map({True: "✓ verified", False: "✗ not exact", None: "—"})
            st.dataframe(checks[["column", "relation", "holds", "reason"]], hide_index=True,
                         alt="Columns excluded from modelling and why",
                         column_config={"reason": st.column_config.TextColumn("reason", width="large")})
            st.markdown("**Model inputs:** " + ", ".join(f"`{c}`" for c in config.MODEL_FEATURES))

    with tabs[4]:
        st.plotly_chart(charts.association_heatmap(profiling.association_matrix(fdf)),
                        alt="Heatmap of pairwise feature associations")
        st.caption("Darker = two columns move together more. Numeric pairs use |Spearman ρ|; pairs "
                   "involving a categorical column use bias-corrected Cramér's V (numeric columns binned "
                   "into quintiles). Strong pairs such as pclass–fare mean the model's importance "
                   "scores split credit between them.")
else:
    with st.container(border=True):
        st.markdown("**Why some columns are not used by the model**")
        st.markdown("- `alive` is just the answer written differently, so using it would be cheating "
                    "(*data leakage*).\n"
                    "- `class`, `who`, `adult_male`, `alone` and `embark_town` repeat information "
                    "already in other columns.\n"
                    f"- `deck` is missing for {df['deck'].isna().mean():.0%} of passengers.")
    detail_hint()
