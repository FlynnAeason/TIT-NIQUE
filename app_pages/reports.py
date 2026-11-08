"""Reports: executive summary, comparison and PDF export."""

import streamlit as st

from src.dashboard import state
from src.reporting.pdf_report import ReportGenerationError, build_pdf
from src.reporting.summary import build_summary, comparison_table, summary_markdown
from src.visualization import charts


@st.cache_data(show_spinner=False, max_entries=2)
def _pdf(generated_at: str, _results: dict) -> bytes:
    """Cache the PDF per experiment run (keyed on its timestamp)."""
    return build_pdf(_results)


st.title("Full report")
st.markdown("The complete written summary of the project, built from the latest results. It is more "
            "technical than the other pages; terms are explained in the **Glossary** in the sidebar.")
st.caption("Re-training the models updates this page and the PDF automatically.")

res = state.require_results()

# Reserve the slot at the top; the slow PDF build runs last so the page renders first.
pdf_slot = st.container(border=True, horizontal=True, vertical_alignment="center")
with pdf_slot:
    st.markdown(f"**PDF model report** · experiment run {res['generated_at'].replace('T', ' ')} UTC")

st.subheader("Executive summary")
st.markdown(summary_markdown(build_summary(res)))

st.subheader("Model comparison")
st.dataframe(comparison_table(res), hide_index=True, alt="Model comparison table")
a, b = st.columns(2)
a.plotly_chart(charts.metric_comparison(res), alt="Test metrics by model")
b.plotly_chart(charts.roc_curves(res, res["selection"]["selected_model"]), alt="ROC curves")

with pdf_slot:
    try:
        with st.spinner("Building PDF…"):
            pdf = _pdf(res["generated_at"], res)
    except ReportGenerationError as exc:
        st.error(str(exc), title="PDF export failed")
    else:
        st.download_button("Download PDF report", pdf, "titanic_model_report.pdf", "application/pdf",
                           type="primary", icon=":material/picture_as_pdf:")
