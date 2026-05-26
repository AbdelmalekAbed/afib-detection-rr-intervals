"""Cross-Dataset — Phase 5 results with patient-level drill-down."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import f1_score

from utils import (
    get_demo_patients_ltafdb,
    header,
    kpi_row,
    load_oof_npz,
    load_results_json,
)

st.set_page_config(page_title="Cross-Dataset", page_icon="🌍", layout="wide")

header(
    "🌍 Cross-Dataset Robustness",
    "Phase 5 — does the AFDB-trained model survive a real shift to LTAFDB? "
    "Four configurations measured under identical 5-fold patient-level CV.",
)

try:
    phase5 = load_results_json("phase5")
except FileNotFoundError:
    st.error("Phase 5 results not found. Run `python -m scripts.phase5_cross_dataset`.")
    st.stop()

cfgs = phase5["configurations"]
ds = phase5["datasets"]

# --------------------------------------------------------------------------- #
# Hero numbers
# --------------------------------------------------------------------------- #
kpi_row(
    [
        ("AFDB internal", f"{cfgs['afdb_internal_fold0']['mean_patient_f1']:.3f}", "fold 0 val (5 patients)"),
        ("LTAFDB zero-shot", f"{cfgs['ltafdb_zero_shot']['mean_patient_f1']:.3f}", "62 patients, never seen"),
        ("LTAFDB scratch", f"{cfgs['ltafdb_scratch_5fold']['mean_patient_f1']:.3f}", "intrinsic LTAFDB ceiling"),
        ("LTAFDB fine-tuned", f"{cfgs['ltafdb_finetuned_5fold']['mean_patient_f1']:.3f}", "AFDB pretrain + LTAFDB FT"),
    ]
)

# --------------------------------------------------------------------------- #
# Configuration comparison bar
# --------------------------------------------------------------------------- #
labels = ["AFDB\ninternal", "LTAFDB\nzero-shot", "LTAFDB\nscratch", "LTAFDB\nfine-tuned"]
keys = ["afdb_internal_fold0", "ltafdb_zero_shot", "ltafdb_scratch_5fold", "ltafdb_finetuned_5fold"]
f1p = [cfgs[k]["mean_patient_f1"] for k in keys]
auroc = [cfgs[k]["auroc"] for k in keys]
f1w = [cfgs[k]["f1"] for k in keys]

bar = go.Figure()
bar.add_trace(go.Bar(name="F1 / patient", x=labels, y=f1p, marker_color="#1f77b4"))
bar.add_trace(go.Bar(name="F1 window-level", x=labels, y=f1w, marker_color="#ff7f0e"))
bar.add_trace(go.Bar(name="AUROC", x=labels, y=auroc, marker_color="#2ca02c"))
bar.update_layout(
    barmode="group", template="plotly_white", height=420,
    yaxis=dict(range=[0, 1.05]),
    title="Three metrics, four configurations — note the gap between window-F1 and patient-F1",
)
st.plotly_chart(bar, use_container_width=True)

st.info(
    "**The two F1 axes tell different stories.** Window-level F1 ≈ 0.95 across the board — "
    "the model classifies individual fenêtres very well. But F1/patient (averaged over patients) "
    "tops out at 0.77 because a few hard patients pull the average down. "
    "This is the real-world clinical metric and it has a ceiling — confirmed by both datasets."
)

# --------------------------------------------------------------------------- #
# Per-patient deep dive
# --------------------------------------------------------------------------- #
header("Per-patient distribution on LTAFDB")

patients = get_demo_patients_ltafdb()
threshold = st.slider("Decision threshold (recomputes per-patient F1)", 0.0, 1.0, 0.5, 0.01)


def patient_f1s(score_key: str):
    out = []
    for pid, p in patients.items():
        y = p["y"]
        if y.sum() == 0 or y.sum() == len(y):
            continue
        s = p[score_key]
        out.append({"patient": pid, "afib_rate": p["afib_rate"], "n": p["n_windows"],
                     "f1": f1_score(y, (s >= threshold).astype(int), zero_division=0)})
    return pd.DataFrame(out)


pf_zs = patient_f1s("zero_shot").assign(config="zero_shot")
pf_sc = patient_f1s("scratch").assign(config="scratch")
pf_ft = patient_f1s("finetuned").assign(config="finetuned")
allf = pd.concat([pf_zs, pf_sc, pf_ft])

fig_box = px.violin(
    allf, x="config", y="f1", box=True, points="all",
    color="config",
    color_discrete_map={"zero_shot": "#d62728", "scratch": "#2ca02c", "finetuned": "#9467bd"},
    title="Per-patient F1 distribution (each dot = one of 62 scorable LTAFDB patients)",
)
fig_box.update_layout(template="plotly_white", height=460, showlegend=False)
st.plotly_chart(fig_box, use_container_width=True)

# --------------------------------------------------------------------------- #
# Scatter: AFib rate vs F1 to spot pathologies
# --------------------------------------------------------------------------- #
header("Where the patient F1 lives — vs AFib rate, vs zero-shot performance")
col_a, col_b = st.columns(2)

with col_a:
    fig_a = px.scatter(
        pf_zs.merge(pf_sc[["patient", "f1"]], on="patient", suffixes=("_zs", "_sc")),
        x="f1_zs", y="f1_sc",
        hover_data=["patient", "afib_rate", "n"],
        title="LTAFDB scratch vs zero-shot, per patient",
        labels={"f1_zs": "F1 (AFDB→LTAFDB zero-shot)", "f1_sc": "F1 (LTAFDB scratch)"},
    )
    fig_a.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                     line=dict(color="black", dash="dash", width=1))
    fig_a.update_layout(template="plotly_white", height=440)
    st.plotly_chart(fig_a, use_container_width=True)
    st.caption("Diagonal = equal performance. Points above = scratch beats zero-shot for that patient.")

with col_b:
    merged = pf_zs.merge(pf_ft[["patient", "f1"]], on="patient", suffixes=("_zs", "_ft"))
    fig_b = px.scatter(
        merged, x="afib_rate", y="f1_zs",
        hover_data=["patient", "n"],
        title="Patient AFib rate vs zero-shot F1",
        labels={"afib_rate": "AFib rate", "f1_zs": "F1 zero-shot"},
        color="f1_zs", color_continuous_scale="RdYlGn",
    )
    fig_b.update_layout(template="plotly_white", height=440)
    st.plotly_chart(fig_b, use_container_width=True)
    st.caption("Patients with extreme AFib rates (~0 or ~1) are excluded automatically by the metric.")

# --------------------------------------------------------------------------- #
# Hardest patients
# --------------------------------------------------------------------------- #
header("Hardest patients — where the ceiling shows up")
worst = pf_sc.sort_values("f1").head(10).copy()
worst = worst.merge(pf_zs[["patient", "f1"]].rename(columns={"f1": "f1_zs"}), on="patient")
worst = worst.merge(pf_ft[["patient", "f1"]].rename(columns={"f1": "f1_ft"}), on="patient")
worst.columns = ["patient", "AFib rate", "n_windows", "F1 scratch", "config", "F1 zero-shot", "F1 finetune"]
worst = worst[["patient", "AFib rate", "n_windows", "F1 zero-shot", "F1 scratch", "F1 finetune"]]
st.dataframe(worst.round(3), use_container_width=True, hide_index=True)
st.caption(
    "These patients pull every configuration down. Likely non-AFib arrhythmias (flutter, AVNRT, "
    "multifocal atrial tachycardia) that look AFib-like under RR-only features. They are the "
    "ceiling of what RR alone can resolve."
)
