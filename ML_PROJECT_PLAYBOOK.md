# ML Portfolio Project Playbook

A reusable brief for turning a notebook or dataset into an honest, understandable, portfolio-grade
machine-learning app. It was distilled from the Titanic project in this folder, including what went
wrong along the way. Copy it into a new project, fill in **Section 0**, and hand it to Claude (or
follow it yourself).

Written for **binary classification**. Notes for regression and multi-class are in Section 12.

---

## 0. Fill in for this project

| Item | Value |
|---|---|
| Project name | `<name>` |
| Starting material | `<path to notebook / script / dataset>` |
| Dataset source | `<URL or file>`, pinned by SHA-256 |
| Target column | `<target>` (positive class = `<label>`) |
| Unit of prediction | one `<passenger / customer / patient / …>` |
| Audience | `<e.g. recruiters + non-technical viewers>` |
| Primary metric | `<ROC-AUC by default; justify if different>` |
| Candidate models | `<e.g. Logistic Regression, Random Forest, Decision Tree, HistGradientBoosting, SVM>` |
| Stack | Python 3.11+, pandas, NumPy, SciPy, scikit-learn, Streamlit, Plotly, Matplotlib, Joblib, ReportLab, pytest |

---

## 1. Ground rules (apply to every step)

1. **Inspect before changing.** Read the whole notebook (markdown, code, outputs) and the real
   data. Never assume a dataset's columns or meaning; verify them.
2. **Never fabricate.** Every number, chart, model name and conclusion is computed from current
   artifacts. If something is missing, show an empty state with the command that fixes it.
3. **No hard-coded conclusions.** Sentences like "X is better than Y" or "women survived more" are
   generated with the comparison in code, so they stay true if results change. *(Titanic: several
   hand-written claims had to be replaced with computed ones during review.)*
4. **The test set is touched once**, after model selection. Never pick a model, threshold or
   feature using test results, even when the test result is disappointing; report the
   disagreement instead.
5. **Plain first, detail on request.** Every page and every number has a one-sentence plain
   version; technical depth sits behind a toggle.
6. **Say what was not verified.** Browser checks, deployment and anything skipped are reported as
   such.
7. **Preserve the original notebook** unchanged in `notebooks/`.

---

## 2. Phase 1 — Inspect and plan

- Profile the data: shape, dtypes, missing values, duplicates, target balance, ranges, outliers.
- **Leakage and redundancy hunt.** For every column, ask "is this the answer in disguise?" and
  "is this a copy or a function of another column?" Verify each suspicion *exactly* in code
  (e.g. `(alive == 'yes') == survived` for all rows). Keep the checks in a function that the app
  and report reuse.
- **Audit the notebook's pipeline for silent bugs.** Typical finds:
  - `select_dtypes(['number'])` / `(['object','category'])` silently drops `bool` columns.
  - CV splitter with `shuffle=True` but no `random_state` → not reproducible.
  - Ordinal-looking categories (class 1/2/3) scaled as numbers.
  - Only one hyperparameter tuned, or the most important one (e.g. `C`) never tuned.
  - Deprecated arguments (check with `-W error::FutureWarning`).
  - Models compared on the test set.
- Write down: kept features, dropped features + reason, engineered features (row-wise only).

## 3. Phase 2 — Project structure

```text
project/
├── app.py                     # Streamlit entry: navigation + sidebar only
├── app_pages/                 # one file per page (not pages/)
├── config.py                  # paths (relative to this file), seed, features, exclusions, palette
├── requirements.txt  README.md  ROADMAP.md  .gitignore  pytest.ini
├── .streamlit/config.toml     # theme (light/dark), no CSS hacks
├── data/raw/  data/processed/ # raw file not committed; split indices saved
├── models/                    # joblib pipelines + metadata.json (hashes, versions)
├── reports/                   # results JSON, test predictions CSV, PDF, logs
├── docs/                      # README screenshots
├── notebooks/                 # original notebook, untouched
├── src/
│   ├── data/          loader (download + hash check), validation (schema), profiling (EDA + leakage checks)
│   ├── features/      row-wise engineering (no learned statistics)
│   ├── preprocessing/ Pipeline + ColumnTransformer
│   ├── training/      model registry, train.py (CLI), multi_seed.py
│   ├── evaluation/    metrics, bootstrap, thresholds, interpretation, baselines
│   ├── prediction/    verified loading, single/batch predict, what-if, trust aids, 3D landscape
│   ├── visualization/ Plotly (app) + Matplotlib (PDF) with one shared palette
│   ├── reporting/     generated executive summary + ReportLab PDF
│   ├── dashboard/     cached loaders, empty states, plain-language helpers, glossary
│   └── utils/         logging, hashing, JSON
└── tests/
```

## 4. Phase 3 — Leakage-safe modelling

- **All learned preprocessing inside one sklearn `Pipeline`** (imputer → scaler / encoder). Feature
  engineering that learns nothing (log, flags, ratios) can run in a `FunctionTransformer` at the front.
- Missing values: median/most-frequent imputation **with a missing indicator** for informative gaps.
- Stratified, seeded train/test split; save the split indices.
- Per model: `RandomizedSearchCV` (stratified K-fold, refit on the primary metric, all metrics
  recorded). Search ranges: log scales for `C`/learning rates; if the best value lands on the edge
  of the range, **widen the range and re-run**.
- **Nested CV** for model selection (outer folds wrap the whole search) so the score reflects the
  tuning procedure, not the luckiest configuration. Ties → lower variance.
- Probabilities for every model. For SVMs use
  `CalibratedClassifierCV(SVC(), method="sigmoid", ensemble=False)`: `SVC(probability=True)` is
  deprecated in scikit-learn 1.9.
- Persist each fitted pipeline with joblib and record its **SHA-256 + library versions** in
  `metadata.json`. Load only `.joblib` files inside `models/` whose hash matches (pickles can run
  code).

## 5. Phase 4 — Honest evaluation

Compute on the untouched test set, after selection:

- Accuracy, precision, recall, specificity, F1, balanced accuracy, ROC-AUC, PR-AUC, MCC, Brier,
  log loss. Undefined metrics become `NaN`, never a silent 0.
- **Stratified bootstrap 95% CIs** for each metric (say they cover test-sample noise only).
- **Paired bootstrap** of the metric difference between the selected model and each alternative.
- Confusion matrix, classification report, ROC and PR curves (PR chance = prevalence, not 0.5).
- Threshold table from **out-of-fold training predictions**, never from the test set.
- Learning curve for the selected model.
- **Permutation importance on held-out data, per raw column** (groups one-hot dummies and
  engineered features). Tree impurity importance and LR coefficients only as secondary views.
  Only logistic-regression coefficients are labelled log-odds (a linear SVM's `coef_` is not).
- **Baseline ladder:** score simple hand-rules on the same test rows ("everyone = majority class",
  one-feature rule, two-feature rule), with rules learned from training rows only. State plainly
  how many more cases the model gets right. *(Titanic: only 3 of 179; the model's real value was
  graded probabilities.)*
- **Multi-seed check** (`multi_seed.py`): repeat split → tune → select → test over ~20 seeds with a
  lighter search. Report each model's mean ± spread, how often each wins, how often the CV pick was
  also the test winner, and where the main split falls in its range. This answers "was it a fluke?".

## 6. Phase 5 — The dashboard (Streamlit)

**Structure:** `st.navigation` with grouped sections in story order:
*Start here* → *The data* → *The model* (how well it works, what it relies on, how it was chosen,
see it in 3D, full report) → *Try it*.

**Simple / Detailed mode:** a sidebar toggle (off by default, `key=` so it persists across pages).
Simple mode shows plain sentences and a few numbers; Detailed mode adds every metric, CI and
technical chart. Nothing is removed, only hidden.

**Plain-language toolkit** (`src/dashboard/explain.py`):
- `in_words(0.799)` → "about 4 out of 5" (denominators 2, 3, 4, 5, 10).
- One sentence per metric ("Finds about 2 out of 3 of the people who actually survived").
- Plain metric labels with the technical term in the tooltip.
- A sidebar glossary (~15 terms), a caption under every chart, and confusion-matrix cells labelled
  "Correct / Missed / False alarm" instead of TP/FP.

**Must-have interactive features:**
1. **"Start here" verdict:** accuracy in words, the majority-class baseline, the trust range (CI),
   the biggest factor, and a 4-step "how it works".
2. **Live what-if predictor:** no submit button; example presets; a "what if one detail changed?"
   curve for each feature; the details that can move this case most.
3. **"Can you trust this answer?"** All-model vote (unanimous / strong / split), plus "what
   happened to similar real cases" (match on the strongest features, loosen until ≥10 found, Wilson
   CI, note that the model learned from those same records).
4. **Cut-off slider** in an `@st.fragment`: live counts and deltas vs. the default, labelled
   "explore only; the deployed cut-off doesn't change".
5. **"Was it a fluke?"** multi-seed box plot with the original split highlighted.
6. **3D landscape:** a rotatable Plotly `Surface` of the model's probability over two numeric
   features, a semi-transparent cut-off plane, real cases above/below, a plain reading, and a flat
   heat map with the decision contour.
7. **Batch CSV** scoring with a template, row-level validation errors and a download.
8. **Full report** page with the generated summary and a PDF download (build it last on the page
   into a reserved `st.container()` so the page doesn't block).

**Streamlit specifics:**
- Cache data with `st.cache_data` and models with `st.cache_resource`, keyed on file mtimes so
  retraining refreshes the app.
- Use `width=` (not `use_container_width`), `alt=` on charts and tables, `title=` on alerts, and
  Material icons.
- Plotly figures: don't hard-code text colours (let the theme handle light/dark); put legends where
  they can't collide with annotations; make `legend` and `margin` overridable in the shared style
  helper.

## 7. Phase 6 — Reporting

- **Executive summary generated from results** (objective, data, data quality, method, models,
  selection rationale, key results, top factors, findings, limitations, next steps). Wording
  branches on the numbers (e.g. "higher than" vs "lower than", "practically tied" when the margin <
  the SD).
- **ReportLab PDF** with real Matplotlib charts (no kaleido dependency), tables, page footer, and
  `KeepTogether` for heading + figure. Base-14 fonts lack symbols such as ★, Δ and ↔, so map them to
  text. Long axis labels wrap.
- **README:** overview, problem, data + quality table, "what changed from the notebook and why",
  methodology, models, real results table, an honest reading of the results, structure, install /
  run commands (one per code block), artifacts, dashboard table, screenshots, limitations, future work.
- **ROADMAP.md:** a guiding rule, what's done (with dates and outcomes), what's next, what's not
  recommended.

## 8. Phase 7 — Testing and verification

- **pytest** covering:
  - data: hash, schema, validation errors listing *every* problem, missing-category handling
  - leakage: excluded columns absent; redundancy claims hold; imputer statistics come from fit
    data only; baseline rules unchanged when test labels are flipped
  - split: disjoint, stratified, reproducible
  - metrics vs. sklearn; fast bootstrap metrics equal the reference; undefined → NaN
  - training end-to-end on a tiny config in `tmp_path`; selection uses CV not test
  - persistence: round-trip, **tampered file rejected**, path traversal rejected
  - single / batch prediction, what-if, vote, similar cases, 3D grid (one row equals the 1-D curve)
  - summary contains the actual numbers; the PDF has ≥ N images
  - **Streamlit `AppTest`:** every page in both modes, mode persists across pages, the slider changes
    counts, presets fill inputs, live prediction changes without a button
- Run with `-W error::FutureWarning` to catch deprecations before they break.
- **Real browser check** of every page, at desktop and phone width; read the server log for errors.
- **Read-through as a first-time user:** cut anything explained twice; replace jargon.

## 9. Screenshots that actually render

Chrome's `--headless --screenshot` captures Streamlit **before it has drawn** (blank or half-drawn
pages), and `--virtual-time-budget` / `--timeout` don't fix it. What works: drive Chrome over the
DevTools protocol (`--remote-debugging-port`, the `websocket-client` package installed in a scratch
folder), then:

1. Navigate and poll until there is no `[data-testid="stStatusWidget"]` and no `stSkeleton`, and
   the main block has text, stable for ~2 s.
2. Wait ~2.5 s more for Plotly (WebGL 3D renders fine headless).
3. Measure `stMainBlockContainer` height, resize the viewport to it, and wait.
4. `Page.captureScreenshot`; use a fresh profile in a temp folder.

## 10. Lessons learned (avoid these)

| Problem | Fix |
|---|---|
| Running Streamlit server keeps an **old version of an edited `src/` module** → `AttributeError: module has no attribute …` | Restart the server after editing shared modules (Ctrl+C, `streamlit run app.py`). Tests in a fresh process are the source of truth. |
| A full browser reload resets session state (e.g. Detailed mode) | Expected; navigate via the menu. Bind to query params if it must survive reloads. |
| A background job's stopped shell keeps its log file locked | Write each new run to a new log file name. |
| Rounding turns 99.6% into "100%" and makes a sentence false | Show ">99%" / "<1%" unless exactly 0 or 1. |
| Chart colours assigned by **rank** change a model's colour between pages | Colour by entity (registration order), never by rank. |
| Text said "four models" after adding a fifth | Count from data; grep for number words after changing the model list. |
| A linear SVM's `coef_` gets shown as log-odds | Restrict coefficient views to logistic regression. |
| Hand-written "insights" go stale or are wrong | Generate every claim from results with explicit comparisons. |
| Confusion matrix shown twice at the default threshold | Merge views or show only one at the default. |
| A best hyperparameter sits on the search-range edge | Widen the range and re-run. |
| The CV winner loses on the single test split | Don't switch models. Report it; run the multi-seed check. *(Titanic: the test split was the selected model's 15th-percentile split, and the top 3–4 models were tied within ±0.03.)* |

## 11. Acceptance checklist

- [ ] Real data loads, is hash-verified and validated; leakage/redundancy decisions verified in code
- [ ] Pipeline is leakage-safe; nested CV selection; test set used once
- [ ] Metrics with CIs, paired comparisons, baseline ladder, multi-seed check
- [ ] Models saved with hashes; tampered or foreign files refused
- [ ] Dashboard: Simple/Detailed, story navigation, glossary, captions, all 8 interactive features
- [ ] Executive summary and PDF generated from current results
- [ ] All tests pass (including AppTest, both modes, `-W error::FutureWarning`)
- [ ] Checked in a real browser (desktop + phone width), server log clean
- [ ] README with real numbers, honest reading, screenshots; ROADMAP up to date
- [ ] Final message lists what was verified and what wasn't

## 12. Adapting to other problem types

- **Regression:** primary metric MAE or RMSE (state why); baseline ladder = predict the mean or
  median, then one-feature rules; replace the confusion matrix and cut-off slider with
  residual-vs-predicted and an error-tolerance slider ("within ±X"); the 3D landscape shows the
  predicted value; the vote shows the spread of predictions.
- **Multi-class:** macro-F1 or balanced accuracy as the primary metric; one-vs-rest ROC; the 3D
  landscape shows the probability of a chosen class; the vote reports the majority class.
- **Imbalanced (<10% positives):** PR-AUC primary; stratify everything; report the baseline
  precision; consider class weights in the search space.
- **Time-ordered data:** replace random splits and K-fold with time-based splits; the multi-seed
  check becomes rolling-origin evaluation.

---

### Commands template

```bash
python -m venv .venv
```

```bash
.venv\Scripts\activate
```

```bash
pip install -r requirements.txt
```

```bash
python -m src.data.loader
```

```bash
python -m src.training.train
```

```bash
python -m src.training.multi_seed
```

```bash
streamlit run app.py
```

```bash
pytest
```
