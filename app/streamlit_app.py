"""Streamlit demo — upload a CSV of RR intervals and inspect AFib predictions.

Launch with: ``make demo`` (or ``streamlit run app/streamlit_app.py``)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="AFib RR Detector", page_icon=":heart:", layout="wide")

st.title("AFib detector from RR intervals")
st.caption("Hybrid CNN-LSTM — research/education only, NOT a medical device.")

with st.sidebar:
    st.header("Settings")
    window_size = st.number_input("Window size (beats)", min_value=10, max_value=120, value=30, step=5)
    threshold = st.slider("Decision threshold", 0.0, 1.0, 0.5, 0.01)
    st.markdown("---")
    st.markdown("**Disclaimer:** this demo is for research and education. Do not use it for medical decisions.")

uploaded = st.file_uploader("Upload a CSV with a single column of RR intervals (in seconds)", type=["csv", "txt"])

if uploaded is None:
    st.info("Upload a file or pick a built-in sample to see predictions.")
    st.stop()

df = pd.read_csv(uploaded, header=None)
rr = df.iloc[:, 0].to_numpy(dtype=np.float32)
st.write(f"Loaded {len(rr)} RR intervals.")

col1, col2 = st.columns(2)
with col1:
    st.subheader("RR series")
    st.line_chart(pd.DataFrame({"rr (s)": rr}))
with col2:
    st.subheader("Poincaré plot")
    if len(rr) > 1:
        st.scatter_chart(pd.DataFrame({"rr_n": rr[:-1], "rr_n+1": rr[1:]}), x="rr_n", y="rr_n+1")

st.subheader("Prediction")
st.warning(
    "Model inference not yet wired in this scaffold. Once a checkpoint is trained "
    "(Phase 3), load it here and apply it window-by-window."
)
