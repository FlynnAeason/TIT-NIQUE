"""Start here: what the project does and how well it works, in plain words."""

import pandas as pd
import streamlit as st

import config
from src.dashboard import state
from src.dashboard.explain import detailed, grade, in_words, ladder_summary, plain_feature
from src.reporting.summary import build_summary, headline
from src.visualization import charts

st.title("Titanic survival prediction")
df = state.require_data()
st.markdown(f"In 1912 the Titanic sank. Of the {len(df):,} passengers in this dataset, "
            f"{in_words(1 - df[config.TARGET].mean())} died. This project teaches a computer **model** "
            "to estimate, from details like age, sex and ticket class, how likely a passenger was to "
            "survive — and explains what it learned.")

res = state.results()
if res is None:
    state.require_results()

h = headline(res)
sel = res["models"][h["key"]]
tm, ci = sel["test_metrics"], sel["test_ci"]
n_test = res["data"]["n_test"]
baseline = 1 - res["data"]["test_positive_rate"]

# ---- The verdict ----------------------------------------------------------
with st.container(border=True):
    st.subheader("The short answer")
    st.markdown(
        f"On **{n_test} passengers the model had never seen**, it was right "
        f"**{in_words(tm['accuracy'])} times** ({tm['accuracy']:.0%}). "
        f"Simply guessing *'did not survive'* for everyone would be right {baseline:.0%} of the time, "
        f"so the model adds real information.")
    if tm["specificity"] > tm["recall"]:
        balance = (f"It is better at ruling out people who died ({in_words(tm['specificity'])}) than "
                   f"at finding people who survived ({in_words(tm['recall'])}).")
    else:
        balance = (f"It finds people who survived ({in_words(tm['recall'])}) at least as well as it "
                   f"rules out people who died ({in_words(tm['specificity'])}).")
    st.markdown(f"{balance} Overall its ranking skill is **{grade(tm['roc_auc'])}** for this kind "
                "of problem.")
    rows = state.ladder()
    if rows is not None:
        st.markdown(ladder_summary(rows)[0] + " See *How well it works* for the full comparison.")
    with st.container(horizontal=True):
        st.metric("Right on new passengers", f"{tm['accuracy']:.0%}", border=True,
                  help="Accuracy on the 179 test passengers.")
        st.metric("Survivors it finds", f"{tm['recall']:.0%}", border=True, help="Recall.")
        st.metric("Non-survivors it rules out", f"{tm['specificity']:.0%}", border=True,
                  help="Specificity.")
        top_feature = sel["permutation_importance"]["values"]["feature"][0]
        st.metric("Biggest factor", plain_feature(top_feature), border=True,
                  help="The detail whose scrambling hurts the model most (permutation importance).")
    st.caption(f"Can you trust these numbers? They come from only {n_test} test passengers, so the "
               f"true accuracy is probably somewhere between {ci['accuracy']['low']:.0%} and "
               f"{ci['accuracy']['high']:.0%}.")

# ---- How it works ---------------------------------------------------------
st.subheader("How it works, in four steps")
steps = [
    (":material/database:", "1 · Collect", f"{len(df):,} real passengers, each with age, sex, ticket "
                                          "class, family aboard, fare and port."),
    (":material/cleaning_services:", "2 · Clean", "Remove columns that would give away the answer "
                                                 "or repeat other columns. Fill in missing ages."),
    (":material/model_training:", "3 · Train & choose", f"Teach {len(res['models'])} different models on "
                                                        "80% of the passengers and keep the one that does "
                                                        "best."),
    (":material/fact_check:", "4 · Test", "Grade the chosen model once on the 20% it never saw. "
                                          "That grade is what you see above."),
]
with st.container(horizontal=True):
    for icon, title, text in steps:
        with st.container(border=True):
            st.markdown(f"{icon} **{title}**")
            st.caption(text)

# ---- What we learned -------------------------------------------------------
sections = {s.heading: s for s in build_summary(res)}
left, right = st.columns([3, 2], gap="large")
with left:
    with st.container(border=True):
        st.markdown("**What the data shows**")
        by = {k: pd.DataFrame(v).set_index(k)["survival_rate"]
              for k, v in res["data"]["profile"]["survival_by"].items()}
        lines = []
        if "sex" in by:
            lines.append(f"**By sex:** {by['sex'].get('female', 0):.0%} of women survived, "
                         f"vs. {by['sex'].get('male', 0):.0%} of men.")
        if "pclass" in by:
            p = by["pclass"]
            lines.append(f"**By ticket class:** {p.get('1', 0):.0%} survived in 1st class, "
                         f"{p.get('2', 0):.0%} in 2nd, {p.get('3', 0):.0%} in 3rd.")
        if "who" in by and "child" in by["who"]:
            lines.append(f"**Children under 16:** {by['who']['child']:.0%} survived, "
                         f"vs. {by['who'].get('man', 0):.0%} of adult men.")
        lines.append("The model learns patterns like these. It cannot tell *why* they happened — "
                     "for example, whether a rule like 'women and children first' caused them.")
        st.markdown("\n".join(f"- {line}" for line in lines))
with right:
    with st.container(border=True):
        st.plotly_chart(charts.target_distribution(df), alt="Bar chart of passengers who did and did not survive")
        st.caption(f"Most passengers in the data did not survive: {1 - df[config.TARGET].mean():.0%} "
                   "died. A useful model has to beat simply guessing that.")

st.markdown("**Where to go next:** see *The data → Who was on board* for survival by group, "
            "*The model → How well it works* for the mistakes it makes, or "
            "*Try it → Predict a passenger* to test it yourself.")

# ---- Detailed mode ---------------------------------------------------------
if detailed():
    st.divider()
    st.subheader("Technical summary")
    with st.container(horizontal=True):
        st.metric("Selected model", h["model"], border=True)
        st.metric("CV ROC-AUC", f"{h['cv_auc']:.3f}", border=True,
                  help=f"± {h['cv_auc_std']:.3f} across {res['selection']['estimate']}. Used for selection.")
        st.metric("Test ROC-AUC", f"{h['test_auc']:.3f}", border=True)
        st.metric("Test balanced acc.", f"{h['test_bal_acc']:.3f}", border=True)
        st.metric("Test MCC", f"{h['test_mcc']:.3f}", border=True)
    with st.container(border=True):
        st.markdown("**Why this model**")
        st.markdown(sections["Models and selection"].paragraphs[0])
        st.caption(config.SELECTION_METRIC_RATIONALE)
        for p in sections["Key results (test set, threshold 0.5)"].paragraphs:
            st.markdown(p)
    with st.container(border=True):
        st.markdown("**Limitations**")
        st.markdown("\n".join(f"- {b}" for b in sections["Limitations"].bullets))
else:
    st.caption(":material/tune: Want the model name, cross-validation scores and limitations? "
               "Switch on **Detailed mode** in the sidebar.")
