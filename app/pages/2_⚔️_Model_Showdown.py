"""Model Showdown — same AFDB patient, every Phase-4 compression variant compared.

Pulls the OOF scores from Phase 4 and Phase 4b. Each variant scored the same
patient (under its assigned fold). Visualizes their probability streams overlaid,
plus per-patient F1 ranking.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import f1_score, roc_auc_score

from utils import get_demo_patients_afdb, header, load_oof_npz

st.set_page_config(page_title="Model Showdown", page_icon="⚔️", layout="wide")

header(
    "⚔️ Model Showdown",
    "Same AFDB patient, every compression variant. See where they agree and where they break.",
)

patients = get_demo_patients_afdb()
if not patients:
    st.error("Phase 4 OOF scores not found. Run `python -m scripts.phase4_compression`.")
    st.stop()

# Also load Phase 4b OOF (archi 30k variants)
oof_b = load_oof_npz("phase4b")

# --------------------------------------------------------------------------- #
# Controls
# --------------------------------------------------------------------------- #
ctrl1, ctrl2 = st.columns([3, 1])

with ctrl1:
    pids = sorted(patients.keys(), key=lambda k: -patients[k]["afib_rate"])
    pid_labels = {
        pid: f"Patient {pid} — AFib rate {patients[pid]['afib_rate']:.0%} — fold {patients[pid]['fold']}"
        for pid in pids
    }
    pid = st.selectbox("AFDB patient", pids, format_func=lambda p: pid_labels[p])

with ctrl2:
    threshold = st.slider("Decision threshold", 0.0, 1.0, 0.5, 0.01)

p = patients[pid]
y = p["y"]
n = len(y)
time_idx = np.arange(n)

available_variants = [k for k in p.keys() if k not in {"y", "fold", "n_windows", "afib_rate"}]
selected = st.multiselect(
    "Variants to compare (architecture 100k params)",
    available_variants,
    default=["fp32", "int8_dynamic", "prune70_finetune", "onnx_fp32"],
)

# Pull matching Phase 4b scores for the same patient
groups_b = oof_b.get("groups", np.array([]))
mask_b = groups_b == pid if len(groups_b) else np.zeros(0, dtype=bool)
phase4b_variants = []
if mask_b.any():
    for k in ["fp32", "int8_dynamic"]:
        if k in oof_b:
            phase4b_variants.append(k)

show_30k = st.checkbox(
    f"Also overlay architecture-30k variants ({', '.join(phase4b_variants)})",
    value=False,
    disabled=not phase4b_variants,
)

# --------------------------------------------------------------------------- #
# Compute per-variant metrics
# --------------------------------------------------------------------------- #
rows = []


def collect_row(name: str, s: np.ndarray, archi: str):
    if y.sum() == 0 or y.sum() == n:
        return None
    y_pred = (s >= threshold).astype(int)
    return {
        "variant": name, "archi": archi,
        "f1": f1_score(y, y_pred, zero_division=0),
        "auroc": roc_auc_score(y, s),
        "afib_preds": int(y_pred.sum()),
    }


for v in selected:
    r = collect_row(v, p[v], "100k")
    if r:
        rows.append(r)

if show_30k:
    for v in phase4b_variants:
        s = oof_b[v][mask_b]
        r = collect_row(f"{v}_p35", s, "30k")
        if r:
            rows.append(r)

if not rows:
    st.warning("Patient has only one class — no F1 to compute.")
    st.stop()

mdf = pd.DataFrame(rows).set_index("variant").sort_values("f1", ascending=False)

st.dataframe(
    mdf.assign(
        f1=mdf["f1"].map(lambda x: f"{x:.4f}"),
        auroc=mdf["auroc"].map(lambda x: f"{x:.4f}"),
    ),
    use_container_width=True,
)

# --------------------------------------------------------------------------- #
# Overlay plot
# --------------------------------------------------------------------------- #
fig = go.Figure()

# Ground truth as filled background
afib_mask = y == 1
if afib_mask.any():
    # Find contiguous runs
    diff = np.diff(np.concatenate([[0], afib_mask.astype(int), [0]]))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    for s, e in zip(starts, ends):
        fig.add_vrect(
            x0=s, x1=e - 0.5,
            fillcolor="#d62728", opacity=0.10, line_width=0,
            layer="below",
        )

palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]
i = 0
for v in selected:
    fig.add_trace(go.Scatter(
        x=time_idx, y=p[v], mode="lines",
        line=dict(width=1.4, color=palette[i % len(palette)]),
        name=f"{v} (100k)",
        hovertemplate=f"<b>{v}</b><br>window %{{x}}<br>score=%{{y:.3f}}<extra></extra>",
    ))
    i += 1

if show_30k:
    for v in phase4b_variants:
        s = oof_b[v][mask_b]
        fig.add_trace(go.Scatter(
            x=time_idx, y=s, mode="lines",
            line=dict(width=1.4, color=palette[i % len(palette)], dash="dot"),
            name=f"{v} (30k)",
            hovertemplate=f"<b>{v}_p35</b><br>window %{{x}}<br>score=%{{y:.3f}}<extra></extra>",
        ))
        i += 1

fig.add_hline(y=threshold, line_dash="dash", line_color="black", annotation_text=f"thr={threshold:.2f}")

fig.update_layout(
    height=480,
    template="plotly_white",
    xaxis_title=f"Window index (red bands = ground truth AFib, {int(afib_mask.sum())}/{n} windows)",
    yaxis_title="AFib probability",
    yaxis=dict(range=[0, 1]),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(l=40, r=20, t=40, b=40),
)

st.plotly_chart(fig, use_container_width=True)

st.caption(
    "Red shaded bands = ground-truth AFib windows. Lines = each model's probability output. "
    "Dotted lines = architecture-30k variants when the toggle is on. Where lines diverge, "
    "the variants disagree on borderline windows — usually short AFib bursts."
)

# --------------------------------------------------------------------------- #
# Variant winner per window
# --------------------------------------------------------------------------- #
header("Where each variant disagrees with FP32 reference")

if "fp32" in selected:
    ref = (p["fp32"] >= threshold).astype(int)
    st.caption(
        f"Reference variant: **fp32** (100k params). Showing the count of windows where each "
        f"other variant predicts something different from FP32, broken down by whether "
        f"the divergence helped (matches GT) or hurt (mismatches GT)."
    )
    rows = []
    for v in selected:
        if v == "fp32":
            continue
        v_pred = (p[v] >= threshold).astype(int)
        diff = v_pred != ref
        helped = int(((v_pred == y) & diff).sum())
        hurt = int(((v_pred != y) & diff).sum())
        rows.append({
            "variant": v,
            "windows different from FP32": int(diff.sum()),
            "of which: helped (matches GT)": helped,
            "of which: hurt (mismatches GT)": hurt,
            "net": helped - hurt,
        })
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
else:
    st.info("Select `fp32` to enable the reference comparison view.")
