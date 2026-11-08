"""See it in 3D: a model's chance of survival over two details at once."""

import numpy as np
import streamlit as st

import config
from src.dashboard import state
from src.dashboard.explain import detailed
from src.prediction.landscape import AXES, matching_passengers, probability_grid
from src.visualization import charts

st.title("See it in 3D")
st.markdown("Pick a kind of passenger and two details. The model's **chance of survival** is drawn as a "
            "landscape: **height and colour = chance**. The grey sheet is the 50% cut-off — where the "
            "landscape rises above it, the model answers *'survived'*. **Drag to rotate, scroll to zoom.**")

res = state.require_results()
df = state.require_data()
models = state.all_models()
if not models:
    state.require_model()
names = {k: m.display_name for k, m in models.items()}
sel_key = res["selection"]["selected_model"]
default_model = "svm" if "svm" in models else sel_key
CLASS_NAMES = {1: "1st", 2: "2nd", 3: "3rd"}
typical_fare = df.groupby("pclass")["fare"].median().round(2).to_dict()

key = st.segmented_control("Model", list(names), default=default_model, required=True, key="td_model",
                           format_func=lambda k: names[k] + (" ★" if k == sel_key else ""))

with st.container(border=True):
    c1, c2, c3 = st.columns([1.1, 1, 1.4])
    sex = c1.segmented_control("Sex", ["female", "male"], default="female", required=True, key="td_sex")
    pclass = c1.segmented_control("Ticket class", [1, 2, 3], default=3, required=True, key="td_pclass",
                                  format_func=CLASS_NAMES.get)
    axis_names = list(AXES)
    x_feat = c2.selectbox("Left–right axis", axis_names, index=0, key="td_x", format_func=lambda a: AXES[a][0])
    y_feat = c2.selectbox("Front–back axis", [a for a in axis_names if a != x_feat], index=0, key="td_y",
                          format_func=lambda a: AXES[a][0])
    embarked = c3.selectbox("Port of embarkation", ["S", "C", "Q"], key="td_port",
                            format_func=lambda p: f"{config.EMBARKED_NAMES[p]} ({p})")
    held = [a for a in axis_names if a not in (x_feat, y_feat)]
    held_defaults = {"age": 30.0, "fare": float(typical_fare[pclass]), "sibsp": 0, "parch": 0}
    held_values = {}
    for a in held:
        is_int = a in ("sibsp", "parch")
        lo, hi = AXES[a][1].min(), AXES[a][1].max()
        held_values[a] = c3.number_input(
            f"{AXES[a][0]} (kept fixed)", int(lo) if is_int else float(lo), int(hi) if is_int else float(hi),
            int(held_defaults[a]) if is_int else float(held_defaults[a]), step=1 if is_int else 1.0,
            key=f"td_hold_{a}_{pclass}" if a == "fare" else f"td_hold_{a}")
    show_people = st.toggle("Show real passengers of this sex and class", value=True, key="td_people",
                            help="Blue dots float above the landscape (survived); red-orange dots sit "
                                 "below it (died). They are placed by their own two values.")

base = {"pclass": pclass, "sex": sex, "embarked": embarked, **held_defaults, **held_values}
model = models[key]
xs, ys, z = probability_grid(model, base, x_feat, y_feat)
people = matching_passengers(df, base, x_feat, y_feat) if show_people else None
jitter = {"sibsp": 0.25, "parch": 0.25}

st.plotly_chart(
    charts.landscape_3d(xs, ys, z, AXES[x_feat][0], AXES[y_feat][0], model.threshold, people,
                        jitter.get(x_feat, 0.0), jitter.get(y_feat, 0.0)),
    alt=f"3D surface of {names[key]}'s chance of survival over {AXES[x_feat][0]} and {AXES[y_feat][0]}",
    config={"displaylogo": False})

# ---- Plain reading of what is on screen --------------------------------------------
who = f"{'women' if sex == 'female' else 'men'} in {CLASS_NAMES[pclass]} class"
share = float((z >= model.threshold).mean())
iy, ix = np.unravel_index(np.argmax(z), z.shape)
jy, jx = np.unravel_index(np.argmin(z), z.shape)
# Don't let rounding turn "almost all" into "all": 0 and 1 only when exactly true.
share_txt = (">99%" if 0.995 <= share < 1 else "<1%" if 0 < share < 0.005 else f"{share:.0%}")
lines = [f"For {who}, {names[key]} gives chances from **{z.min():.0%}** to **{z.max():.0%}** across this map, "
         f"and answers *'survived'* on **{share_txt}** of it."]
if share in (0.0, 1.0):
    answer = "survived" if share == 1.0 else "did not survive"
    short = {"age": "age", "fare": "fare", "sibsp": "siblings/spouses", "parch": "parents/children"}
    lines.append(f"The answer is *'{answer}'* everywhere on this map: for {who}, changing "
                 f"{short[x_feat]} and {short[y_feat]} moves the chance but never across the cut-off. Switch the sex or class to see the other side of the sheet.")
if z.max() - z.min() < 0.1:
    lines.append(f"The landscape is almost flat: for {who}, these two details barely change this model's answer.")
else:
    def spot(feature: str, value: float) -> str:
        return {"age": f"age {value:g}", "fare": f"fare £{value:g}",
                "sibsp": f"{value:g} siblings/spouses", "parch": f"{value:g} parents/children"}[feature]
    lines.append(f"Highest at {spot(x_feat, xs[ix])}, {spot(y_feat, ys[iy])} ({z.max():.0%}); "
                 f"lowest at {spot(x_feat, xs[jx])}, {spot(y_feat, ys[jy])} ({z.min():.0%}).")
if people is not None and len(people):
    lines.append(f"{len(people)} real {who} have both values recorded; "
                 f"{int(people['survived'].sum())} survived ({people['survived'].mean():.0%}).")
st.markdown("\n".join(f"- {line}" for line in lines))

aucs = [res["models"][k]["test_metrics"]["roc_auc"] for k in models if k in res["models"]]
st.info("Smooth slopes come from models that weigh each detail steadily (Logistic Regression, Support "
        "Vector Machine). Steps, cliffs and plateaus come from tree-based models, which split passengers "
        "at yes/no cut points."
        + (f" The shapes differ, yet their test scores are close ({min(aucs):.2f}–{max(aucs):.2f} ranking "
           "skill), because most real passengers sit where the models agree."
           if aucs and max(aucs) - min(aucs) < 0.05 else ""),
        title="Try switching models", icon=":material/360:")

with st.expander("Seen from above (flat map)", icon=":material/map:"):
    st.plotly_chart(charts.landscape_flat(xs, ys, z, AXES[x_feat][0], AXES[y_feat][0], model.threshold),
                    alt="Top-down heat map of the same landscape with the cut-off line")
    st.caption("Blue areas: the model says 'survived'. Red-orange areas: 'did not survive'. The black "
               "line traces the 50% cut-off. If there is no line, the answer is the same everywhere.")

st.caption("Details not on the two axes are held at the values shown above. The surface shows how this "
           "model reacts, not what would really have happened to anyone. Fares above £200 are left out "
           "so the view is not squashed by a few luxury tickets.")

if detailed():
    m = res["models"][key]
    st.divider()
    st.markdown(f"**{names[key]} — tuned settings:** " +
                ", ".join(f"`{k}={v:.4g}`" if isinstance(v, float) else f"`{k}={v}`"
                          for k, v in m["best_params"].items()))
    st.caption(f"Grid: {len(xs)} × {len(ys)} = {z.size} predictions. Cross-validated ROC-AUC "
               f"{m['selection_mean']:.3f}; test ROC-AUC {m['test_metrics']['roc_auc']:.3f}.")
