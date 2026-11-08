"""Headless Streamlit tests: every page renders in both Simple and Detailed mode."""

import pytest
from streamlit.testing.v1 import AppTest

import config
from src.dashboard.explain import in_words, metric_sentence

APP = str(config.PROJECT_ROOT / "app.py")
PAGES = ["overview", "explorer", "laboratory", "evaluation", "explainability", "prediction", "reports",
         "landscape"]

needs_artifacts = pytest.mark.skipif(
    not (config.RESULTS_FILE.exists() and config.BEST_MODEL_FILE.exists()),
    reason="Trained artifacts not present; run `python -m src.training.train` first.")


def _open(page: str, detailed: bool) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=60).run()
    if detailed:
        at.toggle(key="detailed").set_value(True).run()
    at.switch_page(f"app_pages/{page}.py").run()
    return at


@needs_artifacts
@pytest.mark.parametrize("detailed", [False, True], ids=["simple", "detailed"])
@pytest.mark.parametrize("page", PAGES)
def test_page_renders_without_exception(page, detailed):
    at = _open(page, detailed)
    assert not at.exception, [e.value for e in at.exception]


@needs_artifacts
def test_simple_mode_hides_technical_tabs():
    simple, full = _open("explorer", False), _open("explorer", True)
    assert len(simple.tabs) < len(full.tabs)
    assert not any("Associations" in t.label for t in simple.tabs)


@needs_artifacts
def test_mode_persists_across_pages():
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.toggle(key="detailed").set_value(True).run()
    at.switch_page("app_pages/evaluation.py").run()
    assert at.toggle(key="detailed").value is True
    assert any("Technical evaluation" in s.value for s in at.subheader)


def _chance(at: AppTest) -> str:
    return next(m.value for m in at.metric if m.label == "Chance of survival")


@needs_artifacts
def test_live_prediction_updates_without_button():
    at = _open("prediction", False)
    assert not at.exception
    before = _chance(at)  # shown immediately, no submit button
    at.segmented_control(key="wi_sex").set_value("female").run()
    at.segmented_control(key="wi_pclass").set_value(1).run()
    assert not at.exception
    assert _chance(at) != before


@needs_artifacts
def test_preset_buttons_fill_the_inputs():
    at = _open("prediction", False)
    at.button(key="preset_Woman, 1st class, 35").click().run()
    assert at.segmented_control(key="wi_sex").value == "female"
    assert at.segmented_control(key="wi_pclass").value == 1
    assert not at.exception


@needs_artifacts
@pytest.mark.parametrize("feature", ["sex", "pclass", "age", "fare", "sibsp", "parch", "embarked"])
def test_what_if_chart_for_each_detail(feature):
    at = _open("prediction", False)
    at.segmented_control(key="wi_feature").set_value(feature).run()
    assert not at.exception


@needs_artifacts
def test_prediction_shows_vote_and_similar_passengers():
    at = _open("prediction", False)
    text = " ".join(m.value for m in at.markdown)
    assert "What do the other models say?" in text
    assert "What happened to similar real passengers?" in text
    assert "of them survived" in text


@needs_artifacts
def test_evaluation_shows_baseline_ladder():
    at = _open("evaluation", False)
    assert any("simple rule" in m.value for m in at.markdown)


@needs_artifacts
def test_landscape_defaults_to_svm_and_switches():
    at = _open("landscape", False)
    assert at.segmented_control(key="td_model").value == "svm"
    for model in ("hist_gradient_boosting", "logistic_regression"):
        at.segmented_control(key="td_model").set_value(model).run()
        assert not at.exception
    at.selectbox(key="td_x").set_value("sibsp").run()
    at.toggle(key="td_people").set_value(False).run()
    assert not at.exception


@needs_artifacts
def test_threshold_slider_changes_answers():
    at = _open("evaluation", False)
    found = lambda: next(m.value for m in at.metric if m.label == "Survivors found")  # noqa: E731
    at_default = found()
    at.slider(key="ev_threshold").set_value(0.2).run()
    assert not at.exception
    assert found() != at_default


@pytest.mark.parametrize("value, expected", [
    (0.799, "about 4 out of 5"), (0.638, "about 2 out of 3"), (0.9, "about 9 out of 10"),
    (0.5, "about 1 out of 2"), (0.99, "almost all"), (0.01, "almost none"),
])
def test_in_words(value, expected):
    assert in_words(value) == expected


def test_metric_sentence_contains_value():
    assert "80%" in metric_sentence("accuracy", 0.8)
    assert "coin flip" in metric_sentence("roc_auc", 0.81)
