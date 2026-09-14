"""Per-account churn-risk dashboard.

Usage:  streamlit run app.py        (from the repo root)

Loads artifacts/model.joblib + data/curated/saas_customers.csv, lets you pick any
account, and shows its predicted churn probability with a SHAP waterfall for that
single account - the "why" in a form a non-technical VP can read.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src"))

import joblib
import numpy as np
import pandas as pd
import shap
import streamlit as st

from churn.model import make_tree_explainer
from churn.synthesize import FEATURES

st.set_page_config(page_title="SaaS Churn Radar", layout="wide")
st.title("SaaS Churn Radar - account risk explorer")
st.caption(
    "XGBoost churn model + SHAP explanations. Pick an account to see its predicted "
    "renewal risk and exactly which signals drive it."
)

model = joblib.load(os.path.join(HERE, "artifacts", "model.joblib"))
df = pd.read_csv(os.path.join(HERE, "data", "curated", "saas_customers.csv"))

explainer = make_tree_explainer(model)


@st.cache_data
def account_shap(i: int):
    row = df.iloc[[i]][FEATURES]
    sv = np.asarray(explainer.shap_values(row))
    if sv.ndim == 3:
        sv = sv[:, :, 1]
    return sv[0], float(model.predict_proba(row)[0, 1])


col_a, col_b = st.columns([1, 2])
with col_a:
    i = st.selectbox(
        "Account", range(len(df)),
        format_func=lambda k: f"#{k} - {'churned' if df['churned'].iloc[k] else 'renewed'} "
                              f"(MRR ${df['mrr'].iloc[k]:,.0f})",
    )
    sv, prob = account_shap(i)
    st.metric("Predicted churn probability", f"{prob:.1%}")
    actual = "churned" if df["churned"].iloc[i] else "renewed"
    st.write(f"**Actual outcome:** {actual}")

    feats = pd.DataFrame(
        {"feature": FEATURES, "value": [df.iloc[i][f] for f in FEATURES], "shap": sv}
    ).sort_values("shap", key=abs, ascending=False)
    st.dataframe(feats, use_container_width=True, hide_index=True)

with col_b:
    row = df.iloc[[i]][FEATURES]
    # shap 0.49's waterfall takes a single-row Explanation and returns matplotlib Axes;
    # streamlit wants the parent Figure
    _wf = shap.plots.waterfall(explainer(row)[0], show=False)
    st.pyplot(_wf.get_figure(), use_container_width=True)
