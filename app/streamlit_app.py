"""AFib Sandbox — interactive exploration of a CNN-LSTM AFib detector.

This is the entry point. Streamlit auto-discovers the multi-page UI from
``app/pages/*.py``. The Home page here pitches the project and exposes the
key results so a visitor immediately sees what's inside.

Launch:
    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import streamlit as st

from utils import (
    FIGURES,
    PROJECT_ROOT,
    header,
    kpi_row,
    load_compression_csv,
    load_results_json,
)

st.set_page_config(
    page_title="AFib Sandbox",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------- #
# Hero
# --------------------------------------------------------------------------- #
st.markdown(
    """
    # ❤️ AFib Sandbox
    ### Detect atrial fibrillation from RR intervals — explore a research-grade CNN-LSTM, end-to-end.
    """
)
st.caption(
    "Research / education demo. NOT a medical device. "
    "Built from AFDB + LTAFDB (PhysioNet) under patient-level cross-validation."
)

st.divider()

# --------------------------------------------------------------------------- #
# What's inside
# --------------------------------------------------------------------------- #
left, right = st.columns([2, 1])

with left:
    st.markdown(
        """
        ### What this demo shows

        Five connected pages, each surfacing one piece of the work:

        | Page | What you'll see |
        | --- | --- |
        | 🔬 **Patient Inspector** | Pick a real AFDB / LTAFDB patient, see model probability vs ground truth, scrub the timeline |
        | ⚔️ **Model Showdown** | Compare 4 compression variants side-by-side on the same patient |
        | 📦 **Compression Lab** | Interactive Pareto frontier: size vs latency vs F1. Live CPU latency benchmark. |
        | 🌍 **Cross-Dataset** | Phase 5 results — zero-shot AFDB→LTAFDB generalization vs from-scratch ceiling |
        | 🧪 **What-If** | Sketch an RR series by hand, watch the model react in real time |

        Navigate with the sidebar →
        """
    )

with right:
    st.markdown("### Why this matters")
    st.info(
        "Atrial fibrillation is the most common sustained cardiac arrhythmia "
        "(2-4% adult prevalence) and a ~5× risk factor for ischemic stroke. "
        "It's often paroxysmal — easy to miss on a single ECG. "
        "Detection from **RR intervals alone** enables continuous monitoring "
        "on wearables (smartwatches, patches) without a full ECG signal."
    )

st.divider()

# --------------------------------------------------------------------------- #
# Key results
# --------------------------------------------------------------------------- #
header("Key results across the 5 phases")

try:
    phase4 = load_results_json("phase4")
    phase4b = load_results_json("phase4b")
    phase5 = load_results_json("phase5")

    fp32_p4 = phase4["variants"]["fp32"]
    int8_p4b = phase4b["variants"]["int8_dynamic"]
    onnx_p4 = phase4["variants"]["onnx_fp32"]
    ltafdb_zs = phase5["configurations"]["ltafdb_zero_shot"]
    ltafdb_sc = phase5["configurations"]["ltafdb_scratch_5fold"]

    kpi_row(
        [
            ("Best F1 / patient (AFDB)", f"{fp32_p4['mean_patient_f1']:.3f}", "Phase 4 FP32"),
            ("Best F1 / patient (LTAFDB)", f"{ltafdb_sc['mean_patient_f1']:.3f}", "Phase 5 scratch"),
            ("Smallest model", f"{int8_p4b['size_kb_mean']:.0f} KB",
             f"-{(1 - int8_p4b['size_kb_mean'] / fp32_p4['size_kb_mean']) * 100:.0f}% vs FP32 100k"),
            ("Lowest latency (CPU)", f"{onnx_p4['latency_ms_p50'] * 1000:.0f} µs",
             "ONNX FP32, p50"),
        ]
    )

    st.markdown(
        """
        - **F1 / patient plateau ≈ 0.7-0.8** — confirmed independently by AFDB (0.717) and LTAFDB (0.765).
          This is the intrinsic ceiling of *RR-only* features under the patient-level F1 metric.
        - **Zero-shot AFDB → LTAFDB** transfers to **F1/patient = 0.713** without any LTAFDB fine-tuning.
          The model generalizes to a never-seen dataset.
        - **F1 window-level** reaches **0.956 on LTAFDB**, matching the AFib RR-only literature.
        """
    )
except FileNotFoundError as e:
    st.warning(f"Some artifact missing: {e}. Run the phase scripts first.")

st.divider()

# --------------------------------------------------------------------------- #
# Compression Pareto preview
# --------------------------------------------------------------------------- #
header("Compression Pareto — preview")
try:
    df4 = load_compression_csv("phase4")
    df4b = load_compression_csv("phase4b")
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df4["size_kb"], y=df4["mean_patient_f1"],
        mode="markers+text", text=df4["variant"], textposition="top center",
        marker=dict(size=14, color="#1f77b4", line=dict(width=1, color="white")),
        name="Architecture 100k params",
        hovertemplate="<b>%{text}</b><br>Size: %{x:.0f} KB<br>F1/patient: %{y:.3f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=df4b["size_kb"], y=df4b["mean_patient_f1"],
        mode="markers+text", text=df4b["variant"], textposition="bottom center",
        marker=dict(size=14, color="#2ca02c", symbol="diamond", line=dict(width=1, color="white")),
        name="Architecture 30k params",
        hovertemplate="<b>%{text}</b><br>Size: %{x:.0f} KB<br>F1/patient: %{y:.3f}<extra></extra>",
    ))
    fig.add_vline(x=200, line_dash="dash", line_color="#d62728",
                   annotation_text="Plan target: 200 KB", annotation_position="top")
    fig.update_layout(
        height=440,
        xaxis_title="Model size (KB, state_dict serialized)",
        yaxis_title="F1 / patient (5-fold OOF)",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Hover any point for details. The Compression Lab page lets you load each variant "
        "into memory and run a live CPU latency benchmark."
    )
except FileNotFoundError:
    st.info("Run scripts/phase4_compression.py to populate the compression results.")

st.divider()

# --------------------------------------------------------------------------- #
# Reproduce
# --------------------------------------------------------------------------- #
with st.expander("Reproduce all results from scratch"):
    st.code(
        """\
git clone git@github.com:AbdelmalekAbed/afib-detection-rr-intervals.git
cd afib-detection-rr-intervals
make setup                            # uv venv + cpu pytorch + project install
make data                             # download AFDB + LTAFDB + NSRDB from PhysioNet

# Phase 4 + 5 reproduction
python -m scripts.phase4_compression  # ~3 min CPU
python -m scripts.phase4_compression --params-json reports/phase35_best_params.json \\
    --out-prefix phase4b --fig-prefix 17_phase4b --ckpt-tag p35
python -m scripts.phase5_cross_dataset  # ~12 min CPU

streamlit run app/streamlit_app.py    # this app
""",
        language="bash",
    )

st.divider()
st.caption(
    "Built with PyTorch (CPU), ONNX Runtime, Optuna, scikit-learn, Streamlit, Plotly. "
    "All evaluation is patient-level, all comparisons share the same 5-fold split."
)
