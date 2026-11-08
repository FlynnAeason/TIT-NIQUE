"""Titanic survival prediction dashboard.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:  # make `config` and `src` importable from any cwd
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from src.dashboard import state  # noqa: E402
from src.dashboard.explain import GLOSSARY  # noqa: E402

st.set_page_config(page_title="Titanic survival ML", page_icon=":material/sailing:", layout="wide")

PAGES = ROOT / "app_pages"
# Pages are grouped in story order: start -> data -> model -> try it.
page = st.navigation(
    {
        "": [
            st.Page(PAGES / "overview.py", title="Start here", icon=":material/flag:", default=True,
                    help="What this project does, how well the model works, in plain words."),
        ],
        "The data": [
            st.Page(PAGES / "explorer.py", title="Who was on board", icon=":material/groups:",
                    help="Explore the passengers and see which groups survived most."),
        ],
        "The model": [
            st.Page(PAGES / "evaluation.py", title="How well it works", icon=":material/check_circle:",
                    help="How often the model is right on passengers it never saw, and the kinds "
                         "of mistakes it makes."),
            st.Page(PAGES / "explainability.py", title="What it relies on", icon=":material/lightbulb:",
                    help="Which passenger details drive the model's estimates."),
            st.Page(PAGES / "laboratory.py", title="How it was chosen", icon=":material/science:",
                    help="The models we compared and how the winner was picked fairly."),
            st.Page(PAGES / "landscape.py", title="See it in 3D", icon=":material/view_in_ar:",
                    help="Rotate a 3D landscape of any model's chance of survival over two details, "
                         "such as age and fare."),
            st.Page(PAGES / "reports.py", title="Full report", icon=":material/description:",
                    help="Written summary of the whole project, with PDF download."),
        ],
        "Try it": [
            st.Page(PAGES / "prediction.py", title="Predict a passenger", icon=":material/person_search:",
                    help="Describe a passenger and get the model's estimated chance of survival, "
                         "or upload a CSV of many passengers."),
        ],
    },
    position="top",
)

with st.sidebar:
    st.toggle("Detailed mode", key="detailed",
              help="Off: plain-language answers and the key numbers. On: every metric, "
                   "confidence interval and technical chart.")
    st.caption("Simple mode shows the essentials. Turn on Detailed mode for the full analysis.")

    with st.expander("Glossary", icon=":material/menu_book:"):
        for term, meaning in GLOSSARY.items():
            st.markdown(f"**{term}** — {meaning}")

    with st.expander("Project status", icon=":material/info:"):
        res = state.results()
        df = state.data()
        st.badge("Dataset loaded" if df is not None else "Dataset missing",
                 icon=":material/database:", color="green" if df is not None else "red")
        if res is not None:
            sel = res["models"][res["selection"]["selected_model"]]["display_name"]
            st.badge("Models trained", icon=":material/check_circle:", color="green")
            st.caption(f"Model in use: **{sel}**  \nTrained: {res['generated_at'][:10]} · "
                       f"seed {res['config']['random_state']}")
        else:
            st.badge("Models not trained", icon=":material/pending:", color="orange")
            st.code("python -m src.training.train", language="bash")

page.run()
