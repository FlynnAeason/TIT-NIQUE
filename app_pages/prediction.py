"""Try it: live what-if predictor for one passenger, plus batch CSV scoring."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import config
from src.dashboard import state
from src.dashboard.explain import plain_feature
from src.prediction.predict import DataValidationError, predict_frame, predict_one, read_batch_csv
from src.prediction.trust import model_vote, similar_passengers
from src.prediction.whatif import CATEGORICAL, biggest_swing, what_if_curve
from src.visualization import charts

st.title("Predict a passenger")
st.markdown("Describe a passenger and the model's estimated **chance of survival updates instantly**. "
            "Change one detail at a time to see what moves it. It is an estimate based on patterns in "
            "the data — not a fact about any real person.")

model = state.require_model()
df = state.require_data()
CLASS_NAMES = {1: "1st", 2: "2nd", 3: "3rd"}
typical_fare = df.groupby("pclass")["fare"].median().round(2).to_dict()

# ---- Presets (hypothetical passengers) -------------------------------------------
PRESETS = {
    "Man, 3rd class, 30": {"pclass": 3, "sex": "male", "age": 30.0, "sibsp": 0, "parch": 0, "embarked": "S"},
    "Woman, 1st class, 35": {"pclass": 1, "sex": "female", "age": 35.0, "sibsp": 1, "parch": 0, "embarked": "C"},
    "Boy, 2nd class, 6": {"pclass": 2, "sex": "male", "age": 6.0, "sibsp": 1, "parch": 2, "embarked": "S"},
    "Mother, 3rd class, 30": {"pclass": 3, "sex": "female", "age": 30.0, "sibsp": 1, "parch": 3, "embarked": "S"},
    "Man, 1st class, 50": {"pclass": 1, "sex": "male", "age": 50.0, "sibsp": 0, "parch": 0, "embarked": "S"},
}


def _apply_preset(values: dict) -> None:
    for k, v in values.items():
        st.session_state[f"wi_{k}"] = v
    st.session_state["wi_age_known"] = True
    st.session_state["wi_fare"] = float(typical_fare[values["pclass"]])


if "wi_pclass" not in st.session_state:  # first visit: start from the first preset
    _apply_preset(next(iter(PRESETS.values())))

single, batch = st.tabs(["One passenger (live)", "Many passengers (CSV)"])

with single:
    st.markdown("**Try an example**")
    with st.container(horizontal=True):
        for name, values in PRESETS.items():
            st.button(name, on_click=_apply_preset, args=(values,), key=f"preset_{name}")
    st.caption("Examples are made-up passengers. Each uses the typical fare for its ticket class.")

    with st.container(border=True):
        c1, c2, c3 = st.columns(3)
        c1.segmented_control("Ticket class", [1, 2, 3], required=True, key="wi_pclass",
                             format_func=CLASS_NAMES.get)
        c1.segmented_control("Sex", ["female", "male"], required=True, key="wi_sex")
        age_known = c2.toggle("Age known", key="wi_age_known")
        c2.number_input("Age (years)", 0.0, 100.0, step=1.0, key="wi_age", disabled=not age_known)
        c2.number_input("Fare paid (£, 1912)", 0.0, 1000.0, step=1.0, key="wi_fare",
                        help="Typical fare: " + ", ".join(f"{CLASS_NAMES[c]} class £{f:.0f}"
                                                         for c, f in typical_fare.items()) + ".")
        c3.number_input("Siblings / spouses aboard", 0, 10, key="wi_sibsp")
        c3.number_input("Parents / children aboard", 0, 10, key="wi_parch")
        c3.selectbox("Port of embarkation", ["S", "C", "Q"], key="wi_embarked",
                     format_func=lambda p: f"{config.EMBARKED_NAMES[p]} ({p})")

    ss = st.session_state
    passenger = {"pclass": ss.wi_pclass, "sex": ss.wi_sex, "age": ss.wi_age if age_known else None,
                 "sibsp": ss.wi_sibsp, "parch": ss.wi_parch, "fare": ss.wi_fare, "embarked": ss.wi_embarked}
    try:
        out = predict_one(model, passenger)
    except DataValidationError as exc:
        st.error("\n".join(f"- {e}" for e in exc.errors), title="Invalid input")
        st.stop()

    p = out["prob_survived"]
    survived = out["prediction"] == config.POSITIVE_LABEL
    level = ("very likely" if p >= 0.8 else "likely" if p >= 0.6 else "a toss-up" if p > 0.4
             else "unlikely" if p > 0.2 else "very unlikely")
    a, b = st.columns([2, 3], gap="large")
    with a:
        st.metric("Chance of survival", f"{p:.0%}", border=True, help="P(survived) from the model.")
        st.metric("Model's answer", out["prediction"], border=True,
                  help=f"'Survived' when the chance is {model.threshold:.0%} or more.")
    with b:
        fig = go.Figure(go.Bar(
            x=[out["prob_not_survived"], p], y=[config.NEGATIVE_LABEL, config.POSITIVE_LABEL], orientation="h",
            marker_color=[config.CLASS_COLORS[0], config.CLASS_COLORS[1]],
            text=[f"{out['prob_not_survived']:.0%}", f"{p:.0%}"], textposition="auto",
            hovertemplate="%{y}: %{x:.1%}<extra></extra>"))
        fig.add_vline(x=model.threshold, line={"dash": "dash", "color": "#6B7280"})
        fig.update_layout(height=180, margin={"l": 10, "r": 10, "t": 10, "b": 10},
                          xaxis={"range": [0, 1], "tickformat": ".0%"}, showlegend=False)
        st.plotly_chart(fig, alt="Bar chart of the chance of surviving and not surviving")
        st.markdown(f"Survival looks **{level}** for this passenger: a **{p:.0%}** chance, "
                    f"{'above' if survived else 'below'} the {model.threshold:.0%} cut-off.")
        st.caption(("This is close to the cut-off, so the model is unsure — small changes could flip "
                    "the answer. " if abs(p - model.threshold) < 0.1 else "") +
                   ("Age is blank: the model fills in the typical age from its training data. "
                    if not age_known else ""))

    # ---- Can you trust this answer? -------------------------------------------------
    st.subheader("Can you trust this answer?")
    v_col, s_col = st.columns(2, gap="large")
    with v_col:
        with st.container(border=True, height="stretch"):
            st.markdown("**What do the other models say?**")
            vote = model_vote(state.all_models(), passenger)
            if vote["n"] < 2:
                st.caption("Only one verified model is available, so there is nothing to compare with.")
            else:
                yes, n = vote["n_survived"], vote["n"]
                majority_says = config.POSITIVE_LABEL if yes * 2 > n else config.NEGATIVE_LABEL
                if vote["level"] == "unanimous":
                    st.markdown(f":material/check_circle: **All {n} models agree: {majority_says.lower()}.** "
                                f"Their chances range from {vote['low']:.0%} to {vote['high']:.0%}.")
                elif vote["level"] == "strong":
                    st.markdown(f":material/info: **{max(yes, n - yes)} of {n} models say "
                                f"{majority_says.lower()}.** Their chances range from {vote['low']:.0%} "
                                f"to {vote['high']:.0%}.")
                else:
                    st.markdown(f":material/warning: **The models are split {yes}–{n - yes}.** Treat this "
                                "passenger as genuinely uncertain; small details decide the answer.")
                st.dataframe(pd.DataFrame(vote["votes"])[["model", "prob", "survived"]],
                             hide_index=True, alt="Each model's chance of survival for this passenger",
                             column_config={
                                 "model": st.column_config.TextColumn("Model"),
                                 "prob": st.column_config.ProgressColumn("Chance", format="percent",
                                                                         min_value=0, max_value=1),
                                 "survived": st.column_config.CheckboxColumn("Says survived")})
                st.caption(f"The {n} models were built in different ways. When they agree, the answer is "
                           "less likely to be a quirk of one model.")
    with s_col:
        with st.container(border=True, height="stretch"):
            st.markdown("**What happened to similar real passengers?**")
            sim = similar_passengers(df, passenger)
            if sim.n == 0:
                st.caption("No passenger in the data matches this description.")
            else:
                st.markdown(f"In the data there are **{sim.n} {', '.join(sim.criteria)}**. "
                            f"**{sim.survived} of them survived ({sim.rate:.0%}).**")
                lo, hi = sim.ci
                if lo <= p <= hi:
                    compare = "in line with"
                else:
                    compare = "higher than" if p > hi else "lower than"
                st.markdown(f"The model's {p:.0%} is **{compare}** what happened to them "
                            f"(with {sim.n} people, the true rate could be anywhere from {lo:.0%} to {hi:.0%}).")
                with st.expander(f"Show these {sim.n} passengers"):
                    st.dataframe(sim.matches[["survived", "pclass", "sex", "age", "sibsp", "parch", "fare",
                                              "embarked"]], hide_index=True,
                                 alt="Real passengers similar to the one described",
                                 column_config={"survived": st.column_config.CheckboxColumn("survived")})
                st.caption("The model learned from these same records, so this is a sanity check, not an "
                           "independent test." + (" Few matches, so the match was loosened to find enough "
                                                   "people." if len(sim.criteria) <= 2 else ""))

    # ---- What if one detail changed? ---------------------------------------------
    st.subheader("What if one detail were different?")
    st.markdown("Pick a detail. The chart keeps everything else about this passenger the same and "
                "shows how the chance of survival would change.")
    options = ["sex", "pclass", "age", "fare", "sibsp", "parch", "embarked"]
    feature = st.segmented_control("Detail to change", options, format_func=plain_feature, default="age",
                                   required=True, key="wi_feature")
    curve = what_if_curve(model, passenger, feature)
    fmt = {"pclass": CLASS_NAMES.get, "embarked": config.EMBARKED_NAMES.get}.get(feature, str)
    current = passenger[feature]
    st.plotly_chart(charts.what_if_chart(curve, plain_feature(feature), current, model.threshold,
                                         feature in CATEGORICAL, fmt),
                    alt=f"Chance of survival as {plain_feature(feature)} changes")
    lo, hi = curve["prob"].min(), curve["prob"].max()
    st.caption(f"Changing only {plain_feature(feature).lower()} moves the chance between {lo:.0%} and "
               f"{hi:.0%} for this passenger"
               + (" (the orange dot is your passenger)." if feature not in CATEGORICAL and current is not None
                  else " (your passenger's value is in blue).")
               + (" Age is currently blank, so no dot is shown." if feature == "age" and current is None else ""))

    swings = biggest_swing(model, passenger)
    top = [f"**{plain_feature(f)}** ({mn:.0%}–{mx:.0%})" for f, mn, mx in swings[:3]]
    st.markdown("For this passenger, the details that can move the chance the most are " + ", ".join(top) + ".")
    st.caption("These curves show how *this model* reacts for *this passenger*; for someone else the "
               "same change can matter more or less. They describe the model, not what would really "
               "have happened."
               + (" Sudden jumps instead of smooth curves are normal here: tree-based models work with "
                  "yes/no cut points such as 'younger than a certain age'." if model.key != "logistic_regression" else ""))

with batch:
    st.markdown("Upload a CSV file with one passenger per row and the columns "
                + ", ".join(f"`{c}`" for c in config.MODEL_FEATURES) +
                ". `age`, `fare` and `embarked` may be left blank; the other columns are required. "
                "Extra columns are kept in the output. Download the template to see the format.")
    template = pd.DataFrame([
        {"pclass": 1, "sex": "female", "age": 29, "sibsp": 0, "parch": 0, "fare": 211.34, "embarked": "S"},
        {"pclass": 3, "sex": "male", "age": None, "sibsp": 1, "parch": 2, "fare": 23.45, "embarked": "S"},
        {"pclass": 2, "sex": "male", "age": 4, "sibsp": 1, "parch": 1, "fare": 26.0, "embarked": "C"},
    ])
    st.download_button("Download template CSV", template.to_csv(index=False), "passenger_template.csv",
                       "text/csv", icon=":material/download:")
    upload = st.file_uploader("Passenger CSV", type=["csv"], max_upload_size=10)
    if upload is not None:
        try:
            result = predict_frame(model, read_batch_csv(upload))
        except DataValidationError as exc:
            st.error("\n".join(f"- {e}" for e in exc.errors) +
                     "\n\nRow numbers refer to data rows (1 = first row after the header).",
                     title="The file could not be scored")
        else:
            n_surv = (result["prediction"] == config.POSITIVE_LABEL).sum()
            with st.container(horizontal=True):
                st.metric("Rows scored", len(result), border=True)
                st.metric("Predicted survivors", int(n_surv), border=True)
                st.metric("Average chance of survival", f"{result['prob_survived'].mean():.0%}", border=True)
            st.dataframe(result, hide_index=True, alt="Batch predictions",
                         column_config={"prob_survived": st.column_config.ProgressColumn(
                             "Chance of survival", format="percent", min_value=0, max_value=1)})
            st.download_button("Download predictions", result.to_csv(index=False), "predictions.csv",
                               "text/csv", icon=":material/download:", type="primary")
