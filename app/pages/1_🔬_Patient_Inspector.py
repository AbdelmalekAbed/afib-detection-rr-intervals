"""Patient Inspector — pick a real LTAFDB patient and watch the model think.

Shows pre-computed OOF scores from Phase 5 so visualization is instant. For each
patient we plot:
  - Ground-truth AFib labels (window-level, color band)
  - Probability stream from 3 model configurations (zero-shot, scratch, fine-tuned)
  - Per-patient F1 and confusion summary at the chosen threshold

The user can scrub the timeline with a window slider and see which fenêtres
each model gets wrong.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from sklearn.metrics import confusion_matrix, f1_score

from utils import get_demo_patients_ltafdb, header, kpi_row

st.set_page_config(page_title="Patient Inspector", page_icon="🔬", layout="wide")

header(
    "🔬 Patient Inspector",
    "Pick a real LTAFDB patient. See window-by-window predictions from three trained models, "
    "with the ground truth overlaid.",
)

patients = get_demo_patients_ltafdb()
if not patients:
    st.error("Phase 5 OOF scores not found. Run `python -m scripts.phase5_cross_dataset` first.")
    st.stop()

# --------------------------------------------------------------------------- #
# Controls
# --------------------------------------------------------------------------- #
ctrl1, ctrl2, ctrl3 = st.columns([2, 1, 1])

with ctrl1:
    pids_sorted = sorted(patients.keys(), key=lambda k: -patients[k]["afib_rate"])
    pid_labels = {
        pid: f"Patient {pid} — {patients[pid]['n_windows']:,} windows — AFib rate {patients[pid]['afib_rate']:.0%}"
        for pid in pids_sorted
    }
    pid = st.selectbox(
        "Patient", pids_sorted, format_func=lambda p: pid_labels[p], index=0,
    )

with ctrl2:
    threshold = st.slider("Decision threshold", 0.0, 1.0, 0.5, 0.01)

with ctrl3:
    models_to_show = st.multiselect(
        "Models",
        ["zero_shot", "scratch", "finetuned"],
        default=["zero_shot", "scratch", "finetuned"],
        format_func=lambda k: {
            "zero_shot": "AFDB→LTAFDB zero-shot",
            "scratch": "LTAFDB from scratch",
            "finetuned": "AFDB→LTAFDB fine-tuned",
        }[k],
    )

p = patients[pid]
y = p["y"]
n = len(y)
time_idx = np.arange(n)

# --------------------------------------------------------------------------- #
# KPIs
# --------------------------------------------------------------------------- #
metric_rows = []
for key in ["zero_shot", "scratch", "finetuned"]:
    s = p[key]
    y_pred = (s >= threshold).astype(int)
    if y.sum() == 0 or y.sum() == n:
        metric_rows.append({"model": key, "f1": float("nan"), "sens": float("nan"), "spec": float("nan"), "auroc": float("nan")})
        continue
    tn, fp, fn, tp = confusion_matrix(y, y_pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else 0
    spec = tn / (tn + fp) if (tn + fp) else 0
    try:
        from sklearn.metrics import roc_auc_score
        auroc = roc_auc_score(y, s)
    except Exception:
        auroc = float("nan")
    metric_rows.append({
        "model": key, "f1": f1_score(y, y_pred, zero_division=0),
        "sens": sens, "spec": spec, "auroc": auroc,
    })

mdf = pd.DataFrame(metric_rows).set_index("model")

best_model = mdf["f1"].idxmax() if not mdf["f1"].isna().all() else "—"

kpi_row(
    [
        ("Patient", str(pid), f"fold {p['fold']}"),
        ("Windows", f"{n:,}", f"AFib rate {p['afib_rate']:.1%}"),
        ("Best F1 here", f"{mdf['f1'].max():.3f}" if not mdf["f1"].isna().all() else "—", best_model),
        ("Threshold", f"{threshold:.2f}", None),
    ]
)

# --------------------------------------------------------------------------- #
# Main figure — multi-row Plotly: ground truth band + probability streams
# --------------------------------------------------------------------------- #
n_rows = 1 + len(models_to_show)
row_titles = ["Ground truth AFib (per window)"] + [
    {"zero_shot": "AFDB→LTAFDB zero-shot probability",
     "scratch": "LTAFDB from-scratch probability",
     "finetuned": "AFDB→LTAFDB fine-tuned probability"}[m]
    for m in models_to_show
]

fig = make_subplots(
    rows=n_rows, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.05,
    row_heights=[0.18] + [0.82 / max(len(models_to_show), 1)] * len(models_to_show),
    subplot_titles=row_titles,
)

# Ground truth as a heatmap-like band
fig.add_trace(
    go.Heatmap(
        z=[y],
        x=time_idx,
        colorscale=[[0, "#d4e9d2"], [1, "#d62728"]],
        showscale=False,
        zmin=0, zmax=1,
        hovertemplate="window %{x}<br>AFib=%{z}<extra></extra>",
    ),
    row=1, col=1,
)

model_colors = {"zero_shot": "#d62728", "scratch": "#2ca02c", "finetuned": "#9467bd"}
for i, key in enumerate(models_to_show, start=2):
    s = p[key]
    fig.add_trace(
        go.Scatter(
            x=time_idx, y=s, mode="lines",
            line=dict(color=model_colors[key], width=1.2),
            name=key, showlegend=False,
            hovertemplate=f"window %{{x}}<br>{key} score=%{{y:.3f}}<extra></extra>",
        ),
        row=i, col=1,
    )
    fig.add_hline(y=threshold, line_dash="dash", line_color="black", line_width=1, row=i, col=1)
    fig.update_yaxes(range=[0, 1], row=i, col=1)

fig.update_yaxes(showticklabels=False, row=1, col=1)
fig.update_xaxes(title_text="Window index (each window = 60 RR intervals, stride 30)", row=n_rows, col=1)
fig.update_layout(
    height=180 + 220 * len(models_to_show),
    template="plotly_white",
    margin=dict(l=40, r=20, t=50, b=40),
)

st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------------------------------- #
# Per-model table
# --------------------------------------------------------------------------- #
header("Per-model summary at this threshold")
display = mdf.copy()
display["F1"] = display["f1"].map(lambda x: f"{x:.3f}" if not pd.isna(x) else "—")
display["Sensitivity"] = display["sens"].map(lambda x: f"{x:.3f}" if not pd.isna(x) else "—")
display["Specificity"] = display["spec"].map(lambda x: f"{x:.3f}" if not pd.isna(x) else "—")
display["AUROC"] = display["auroc"].map(lambda x: f"{x:.3f}" if not pd.isna(x) else "—")
display.index = display.index.map({
    "zero_shot": "AFDB → LTAFDB (zero-shot)",
    "scratch": "LTAFDB from scratch",
    "finetuned": "AFDB → LTAFDB (fine-tuned)",
})
st.dataframe(display[["F1", "Sensitivity", "Specificity", "AUROC"]], use_container_width=True)

# --------------------------------------------------------------------------- #
# Where the models disagree
# --------------------------------------------------------------------------- #
header("Disagreement zoom")
st.caption(
    "Windows where the three models disagree on the decision (after thresholding). "
    "These are the borderline cases — often correspond to short AFib bursts or "
    "non-AFib arrhythmias that look like AFib under the RR-only feature set."
)

preds = np.stack([
    (p["zero_shot"] >= threshold).astype(int),
    (p["scratch"] >= threshold).astype(int),
    (p["finetuned"] >= threshold).astype(int),
])
disagree = preds.std(axis=0) > 0
n_disagree = int(disagree.sum())
st.metric("Windows with model disagreement", f"{n_disagree:,}", f"{n_disagree / n:.1%} of all windows")

if n_disagree > 0:
    rows = []
    idxs = np.where(disagree)[0]
    for w in idxs[:30]:
        rows.append({
            "window": int(w),
            "ground truth": "AFib" if y[w] == 1 else "Normal",
            "zero-shot": f"{p['zero_shot'][w]:.2f}",
            "scratch": f"{p['scratch'][w]:.2f}",
            "fine-tuned": f"{p['finetuned'][w]:.2f}",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    if len(idxs) > 30:
        st.caption(f"Showing first 30 of {len(idxs)} disagreement windows.")
