# Roadmap: making the project simple to understand

Status: **Steps 1, 2 and 3 are done** — 2026-11-06, 81 tests passing. The "Also good" ideas in
section 3 and everything from section 4 onward are still planned.

## Guiding rule

**Simple answer first, detail on request.** Every new feature shows one plain sentence up front
and puts the technical detail (confidence intervals, nested CV, bootstrap) behind a toggle or
expander. Nothing existing is removed.

---

## 1. Simplify the interface ✅ done

Built as: a **Detailed mode** switch in the sidebar (off by default, remembered across pages);
navigation grouped as *Start here* / *The data* / *The model* / *Try it*; plain-language helpers in
`src/dashboard/explain.py` (numbers as "about 4 out of 5", a 16-term glossary); simple versions of
every page with the technical content kept under Detailed mode. The read-through removed duplicated
percentages and replaced TP/FP wording in the confusion matrix with "missed survivor" / "false alarm".
Known limit: reloading the browser page (not using the menu) resets to Simple mode.

| Change | What it means |
|---|---|
| **Simple / Detailed mode** | Simple (default): plain sentences and a few big numbers. Detailed: today's full metrics. |
| **Story-order navigation** | Regroup 7 pages into 4: *Start here* → *The data* → *The model* → *Try it*. Laboratory, Evaluation, Explainability and Reports move under *The model*. |
| **Every number gets a sentence** | "Accuracy 79.9%" → "Right about 4 out of 5 times." "Recall 63.8%" → "Finds about 2 in 3 of the people who actually survived." |
| **Jargon rules** | Plain words by default, the technical term in brackets, a sidebar glossary, `?` tooltips on anything technical. |
| **Chart captions** | One line under every chart: what to look at and what it means. |
| **Final read-through** | Review the finished app as a first-time user; cut anything that needs explaining twice. |

## 2. Core additions ✅ done

Built as:
- **What-if predictor** (*Try it*): no Predict button; five example passengers; a "What if one detail
  were different?" chart for each of the 7 details; a line naming the details that move this
  passenger's chance most. Code: `src/prediction/whatif.py`.
- **Threshold slider** (*How well it works*): runs in its own fragment so only that section updates;
  shows accuracy, survivors found and false alarms with the change vs. 50%, plus the chart and box
  view at the new cut-off. Clearly labelled as exploration; the model keeps 50%.
- **"Start here" verdict**: already delivered in step 1.
- **Multi-seed evaluation**: `python -m src.training.multi_seed` (20 seeds starting at 42, 10
  candidates per model, non-nested to keep it to ~4 min). Results in
  `reports/multi_seed_results.json`, shown under *How it was chosen*. Finding: average test ROC-AUC
  Random Forest 0.859, Hist Gradient Boosting 0.853, Logistic Regression 0.852, Decision Tree 0.834;
  split-to-split wobble about ±0.03, so the top three are effectively tied. The original split was one
  of Hist Gradient Boosting's weakest (0.815; its range 0.765–0.907).

Original plan:

1. **Live what-if predictor** (Prediction page). Inputs update the result instantly, with a
   "vary one feature" chart (e.g. drag age 0 to 80) and "Try an example" presets.
2. **Threshold slider** (Evaluation page). Move the cut-off and watch the confusion matrix,
   precision and recall update. Includes a one-line example of the trade-off.
3. **"Start here" verdict** (Overview). A plain-English summary such as "Right about 4 out of 5
   times; better at spotting who did not survive than who did."
4. **Multi-seed evaluation.** One script repeating split + nested CV over about 20 seeds (about
   15-25 min), result saved to `reports/`. Answers whether Hist Gradient Boosting's weaker test
   score was just split noise. Run once and cached, never on app load.

## 3. Teaching and trust ideas

**Recommended — ✅ done.** Built as:
- **Baseline ladder** (`src/evaluation/baselines.py`) on *How well it works*, with one line on
  *Start here*. Rules are learned from training passengers only (majority outcome per group) and
  scored on the same test passengers. Result on seed 42: everyone dies 61% · women survive 78% ·
  women and children 78% · sex + class 78% · model 80% (143 vs. 140 of 179). The page says plainly
  that the model adds only 3 passengers at a yes/no level, and that its real extra is a graded
  chance per person.
- **Four-model vote** and **similar passengers** (`src/prediction/trust.py`) on *Predict a
  passenger*, under "Can you trust this answer?". Similar passengers always match sex and class,
  then age within 5 years and alone/with family, loosening until at least 10 are found; the model's
  chance is compared with their Wilson interval.

Original notes:

- **Baseline ladder.** Accuracy at each step: "everyone dies" (61.5%) → "women survive, men
  don't" → "sex + class rules" → our model. Shows how much the model really adds.
- **Four-model vote.** For each prediction, show what all four saved models say; flag a split
  vote as uncertain. No new training needed.
- **Similar passengers.** Next to a prediction: "in the data, 38 passengers like this, 31
  survived (82%)". Grounds the number in real history.

**Also good:**

- **Leakage demo.** A button that trains with `alive` included (100% accuracy) and explains why
  that is cheating.
- **Guess before the model.** Show a real passenger, let the user guess, then reveal the actual
  outcome and the model's guess.
- **Confidence badge** (low / medium / high) from distance to the cut-off.
- **One-page prediction PDF** for a single passenger.
- **Session history** table of tried predictions, downloadable as CSV.
- **Missing-age demo.** Prediction with age known vs. left blank.

## 4. Interaction backlog

- Click-to-filter on charts (click "Female" or "Class 3" to filter the rest)
- Compare 2-3 hypothetical passengers side by side
- Click a confusion-matrix cell to list those passengers
- Model comparison picker (any two models side by side)
- Shareable links (filters and model choice stored in the URL)
- Reset-filters button and saved presets (e.g. "women in 3rd class")

## 5. ML depth backlog

- Probability calibration (calibration curve, isotonic / Platt, Brier before and after)
- SHAP-style reasons for each prediction (optional extra dependency)
- ✅ 3D probability landscape — done 2026-11-07. New page *The model → See it in 3D*
  (`app_pages/landscape.py`, `src/prediction/landscape.py`): rotatable surface of any model's chance
  over two of age / fare / siblings / parents, with the 50% cut-off sheet, real passengers of the same
  sex and class, a plain reading, and a flat top-down map.
- ✅ SVM as a fifth model — done 2026-11-06. Uses `CalibratedClassifierCV(SVC(), ensemble=False)`
  because `SVC(probability=True)` is deprecated in scikit-learn 1.9. Nested-CV ROC-AUC 0.860, test
  0.842; average 0.849 across 20 splits. It did not change the winner. Its weights are not shown as
  log-odds coefficients.
- Fairness check (error rates by sex and class)
- Name-based features (title, family and ticket-group size). Needs the full Kaggle `train.csv`;
  the seaborn copy has no names.

## 6. Portfolio and engineering backlog

- ✅ Screenshots in the README — done 2026-11-06: six pages in `docs/`, captured from the running app
  with headless Chrome.
- Deploy to Streamlit Community Cloud (live link)
- GitHub Actions CI running `pytest`, plus a badge
- Dockerfile

## Not recommended

Deep learning, a FastAPI layer, AutoML, animations, a chatbot, custom JavaScript components.
They add complexity without teaching anything new on 891 rows.

## Suggested build order

1. Simple / Detailed mode and story-order navigation (section 1)
2. What-if predictor, threshold slider, "Start here" verdict (section 2, items 1-3)
3. Baseline ladder, four-model vote, similar passengers (section 3)
4. Multi-seed evaluation (section 2, item 4), the only step needing heavy compute
5. Screenshots, then deployment (section 6)
