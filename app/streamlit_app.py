"""Interactive credit risk demo.

Run from the project root:  streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.predict import load_metadata, score  # noqa: E402

st.set_page_config(page_title="Credit Scoring Demo", page_icon="💳", layout="centered")

meta = load_metadata()
MODELS = {f"{meta['model']} (recommended)": "best", "Neural Network (Keras)": "nn"}
BAND_STYLE = {"Low": ("🟢", st.success), "Medium": ("🟠", st.warning), "High": ("🔴", st.error)}

st.title("💳 Credit Scoring Demo")
st.caption(
    "Estimates the probability that a borrower experiences a serious delinquency "
    "(90+ days past due) within the next two years. Trained on Kaggle's *Give Me Some Credit* data."
)

model_label = st.sidebar.radio("Model", list(MODELS))
kind = MODELS[model_label]
threshold = meta["nn_threshold"] if kind == "nn" else meta["threshold"]
test_auc = meta["nn_test_roc_auc"] if kind == "nn" else meta["test_roc_auc"]
st.sidebar.metric("Test ROC-AUC", f"{test_auc:.3f}")
st.sidebar.metric("Decision threshold", f"{threshold:.2f}")
st.sidebar.caption("Applicants at or above the threshold are flagged as likely to default.")

single, batch = st.tabs(["Single applicant", "Batch scoring (CSV)"])

with single:
    with st.form("applicant"):
        c1, c2 = st.columns(2)
        age = c1.number_input("Age", 18, 110, 45)
        dependents = c2.number_input("Number of dependents", 0, 20, 1)
        monthly_inc = c1.number_input("Monthly income ($)", 0, 1_000_000, 5_000, step=500)
        debt_ratio = c2.number_input(
            "Debt ratio", 0.0, 100.0, 0.35, step=0.05,
            help="Monthly debt payments, alimony and living costs divided by monthly gross income.",
        )
        rev_util = c1.number_input(
            "Revolving credit utilisation", 0.0, 10.0, 0.30, step=0.05,
            help="Total balance on credit cards and personal lines of credit divided by the sum of credit limits.",
        )
        open_credit = c2.number_input("Open credit lines and loans", 0, 100, 8)
        real_estate = c1.number_input("Real estate loans or lines", 0, 50, 1)
        st.markdown("**Payment history (last 2 years)**")
        c3, c4, c5 = st.columns(3)
        late_30_59 = c3.number_input("30-59 days late", 0, 50, 0)
        late_60_89 = c4.number_input("60-89 days late", 0, 50, 0)
        late_90 = c5.number_input("90+ days late", 0, 50, 0)
        submitted = st.form_submit_button("Assess risk", type="primary", width="stretch")

    if submitted:
        applicant = pd.DataFrame([{
            "rev_util": rev_util, "age": age, "late_30_59": late_30_59, "debt_ratio": debt_ratio,
            "monthly_inc": monthly_inc, "open_credit": open_credit, "late_90": late_90,
            "real_estate": real_estate, "late_60_89": late_60_89, "dependents": dependents,
        }])
        result = score(applicant, kind).iloc[0]
        proba, band = result["default_probability"], result["risk_band"]
        icon, box = BAND_STYLE[band]

        m1, m2 = st.columns(2)
        m1.metric("Probability of default", f"{proba:.1%}")
        m2.metric("Risk band", f"{icon} {band}")
        st.progress(min(float(proba), 1.0))
        if result["predicted_default"]:
            box(f"Probability is above the {threshold:.0%} threshold: this applicant is flagged as high risk.")
        else:
            box(f"Probability is below the {threshold:.0%} threshold: this applicant is not flagged.")
        st.caption(
            f"For reference, the average default rate in the data is {meta['test_default_rate']:.1%}. "
            "This is a portfolio project, not a real lending decision tool."
        )

with batch:
    st.write(
        "Upload a CSV with the 10 raw columns (`" + "`, `".join(meta["raw_features"]) + "`) "
        "or the original Kaggle column names."
    )
    file = st.file_uploader("CSV file", type="csv")
    if file is not None:
        df = pd.read_csv(file)
        try:
            scored = pd.concat([df, score(df, kind)], axis=1)
        except ValueError as e:
            st.error(str(e))
        else:
            st.bar_chart(scored["risk_band"].value_counts().reindex(["Low", "Medium", "High"]).fillna(0))
            st.dataframe(scored.head(200), width="stretch")
            st.download_button(
                "Download scored CSV", scored.to_csv(index=False), "scored_applicants.csv", "text/csv"
            )
