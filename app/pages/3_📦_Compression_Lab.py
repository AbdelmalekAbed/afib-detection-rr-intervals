"""Compression Lab — interactive Pareto + live CPU latency benchmark.

Lets the user:
  1. See the Pareto frontier across all 8 + 8 = 16 variants (archi 100k + archi 30k)
  2. Filter by latency budget / size budget and watch the frontier update
  3. Load any specific variant and benchmark its CPU inference latency LIVE
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils import benchmark_latency, header, load_compression_csv, load_variant_model, state_dict_size_kb

st.set_page_config(page_title="Compression Lab", page_icon="📦", layout="wide")

header(
    "📦 Compression Lab",
    "16 model variants (8 compression strategies × 2 architectures). Explore the Pareto frontier and "
    "run a fresh CPU latency benchmark on demand.",
)

# --------------------------------------------------------------------------- #
# Load data
# --------------------------------------------------------------------------- #
try:
    df4 = load_compression_csv("phase4").assign(archi="100k", n_params=100_193)
    df4b = load_compression_csv("phase4b").assign(archi="30k", n_params=30_657)
except FileNotFoundError:
    st.error("Compression CSVs not found. Run scripts/phase4_compression.py first.")
    st.stop()

df = pd.concat([df4, df4b], ignore_index=True)
df["label"] = df["variant"] + " (" + df["archi"] + ")"

# --------------------------------------------------------------------------- #
# Constraint sliders
# --------------------------------------------------------------------------- #
left, right = st.columns([1, 1])

with left:
    size_max = st.slider(
        "Max model size (KB)",
        min_value=int(df["size_kb"].min()) - 5,
        max_value=int(df["size_kb"].max()) + 50,
        value=int(df["size_kb"].max()) + 50,
        step=10,
    )

with right:
    lat_max = st.slider(
        "Max latency (ms, p50)",
        min_value=float(df["latency_ms_p50"].min()),
        max_value=float(df["latency_ms_p50"].max()) + 0.5,
        value=float(df["latency_ms_p50"].max()) + 0.5,
        step=0.05,
        format="%.2f",
    )

eligible = df[(df["size_kb"] <= size_max) & (df["latency_ms_p50"] <= lat_max)].copy()
ineligible = df[~((df["size_kb"] <= size_max) & (df["latency_ms_p50"] <= lat_max))].copy()

best_under_budget = eligible.sort_values("mean_patient_f1", ascending=False).head(1)

if not best_under_budget.empty:
    b = best_under_budget.iloc[0]
    st.success(
        f"**Best variant under your budget**: `{b['variant']}` ({b['archi']} params) — "
        f"F1/patient {b['mean_patient_f1']:.4f}, size {b['size_kb']:.1f} KB, "
        f"latency {b['latency_ms_p50']:.2f} ms p50."
    )
else:
    st.warning("No variant fits this budget. Loosen one of the constraints.")

# --------------------------------------------------------------------------- #
# 3D Pareto plot
# --------------------------------------------------------------------------- #
header("Three-axis Pareto — size × latency × F1/patient")

fig = go.Figure()
for archi, color in [("100k", "#1f77b4"), ("30k", "#2ca02c")]:
    sub = df[df["archi"] == archi]
    fig.add_trace(go.Scatter3d(
        x=sub["size_kb"],
        y=sub["latency_ms_p50"],
        z=sub["mean_patient_f1"],
        mode="markers+text",
        text=sub["variant"],
        textposition="top center",
        marker=dict(
            size=10,
            color=color,
            opacity=0.85,
            line=dict(width=1, color="white"),
        ),
        name=f"Architecture {archi} params",
        hovertemplate=(
            "<b>%{text}</b> (" + archi + ")"
            "<br>Size: %{x:.1f} KB"
            "<br>Latency p50: %{y:.2f} ms"
            "<br>F1/patient: %{z:.4f}<extra></extra>"
        ),
    ))
fig.update_layout(
    scene=dict(
        xaxis_title="Size (KB)",
        yaxis_title="Latency p50 (ms)",
        zaxis_title="F1 / patient",
    ),
    height=560,
    margin=dict(l=0, r=0, t=0, b=0),
)
st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------------------------------- #
# 2D views and budget filter
# --------------------------------------------------------------------------- #
header("Pareto cuts under your budget")

col_a, col_b = st.columns(2)

with col_a:
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=eligible["size_kb"], y=eligible["mean_patient_f1"],
        mode="markers+text", text=eligible["label"], textposition="top center",
        marker=dict(size=12, color=eligible["latency_ms_p50"], colorscale="Viridis",
                    showscale=True, colorbar=dict(title="lat p50 (ms)")),
        name="under budget",
        hovertemplate="<b>%{text}</b><br>Size: %{x:.0f}KB<br>F1: %{y:.3f}<extra></extra>",
    ))
    if not ineligible.empty:
        fig2.add_trace(go.Scatter(
            x=ineligible["size_kb"], y=ineligible["mean_patient_f1"],
            mode="markers", marker=dict(size=8, color="lightgray", opacity=0.5),
            name="outside budget",
            hovertemplate="<b>%{text}</b><br>Size: %{x:.0f}KB<br>F1: %{y:.3f}<extra></extra>",
            text=ineligible["label"],
        ))
    fig2.add_vline(x=size_max, line_dash="dash", line_color="#d62728")
    fig2.update_layout(
        title="Size vs F1/patient",
        xaxis_title="Size (KB)", yaxis_title="F1/patient",
        template="plotly_white", height=420,
        legend=dict(orientation="h", yanchor="bottom", y=-0.35),
    )
    st.plotly_chart(fig2, use_container_width=True)

with col_b:
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=eligible["latency_ms_p50"], y=eligible["mean_patient_f1"],
        mode="markers+text", text=eligible["label"], textposition="top center",
        marker=dict(size=12, color=eligible["size_kb"], colorscale="Plasma",
                    showscale=True, colorbar=dict(title="size (KB)")),
        name="under budget",
        hovertemplate="<b>%{text}</b><br>Lat: %{x:.2f}ms<br>F1: %{y:.3f}<extra></extra>",
    ))
    if not ineligible.empty:
        fig3.add_trace(go.Scatter(
            x=ineligible["latency_ms_p50"], y=ineligible["mean_patient_f1"],
            mode="markers", marker=dict(size=8, color="lightgray", opacity=0.5),
            name="outside budget",
            text=ineligible["label"],
        ))
    fig3.add_vline(x=lat_max, line_dash="dash", line_color="#d62728")
    fig3.update_layout(
        title="Latency vs F1/patient",
        xaxis_title="Latency p50 (ms)", yaxis_title="F1/patient",
        template="plotly_white", height=420,
        legend=dict(orientation="h", yanchor="bottom", y=-0.35),
    )
    st.plotly_chart(fig3, use_container_width=True)

# --------------------------------------------------------------------------- #
# Full table
# --------------------------------------------------------------------------- #
header("Full results table")
show = df.copy()
show = show[["archi", "variant", "size_kb", "size_ratio_vs_fp32", "latency_ms_p50",
              "latency_ms_p95", "sparsity", "mean_patient_f1", "auroc", "f1"]]
show.columns = ["Archi", "Variant", "Size (KB)", "Size ratio vs FP32", "Latency p50 (ms)",
                 "Latency p95 (ms)", "Sparsity", "F1/patient", "AUROC", "F1 window"]
st.dataframe(show.round(4), use_container_width=True, hide_index=True)

# --------------------------------------------------------------------------- #
# LIVE benchmark
# --------------------------------------------------------------------------- #
st.divider()
header("🔥 Live CPU latency benchmark", "Load a 100k-param variant and time it on this machine right now.")

bench_col1, bench_col2 = st.columns([2, 1])
with bench_col1:
    variant_choice = st.selectbox(
        "Variant to benchmark (architecture 100k)",
        ["fp32", "int8_dynamic", "prune30", "prune50", "prune70",
         "prune50_int8", "prune70_finetune"],
    )
with bench_col2:
    n_iters = st.selectbox("Iterations", [50, 100, 200, 500], index=1)

if st.button("Run benchmark", type="primary"):
    with st.spinner(f"Loading {variant_choice} + warming up..."):
        try:
            model = load_variant_model(variant_choice)
            size_kb = state_dict_size_kb(model)
            res = benchmark_latency(model, n_iters=int(n_iters))
        except FileNotFoundError as e:
            st.error(f"Checkpoint not found: {e}")
            st.stop()
        except Exception as e:
            st.error(f"Benchmark failed: {e}")
            st.stop()

    st.success(
        f"**{variant_choice}** — size {size_kb:.1f} KB, "
        f"p50={res['p50']:.3f} ms, p95={res['p95']:.3f} ms, "
        f"mean={res['mean']:.3f} ms (n={n_iters})"
    )

    raw = np.array(res["raw_ms"])
    hist = go.Figure()
    hist.add_trace(go.Histogram(x=raw, nbinsx=40, marker=dict(color="#1f77b4", line=dict(color="white", width=0.5))))
    hist.add_vline(x=res["p50"], line_dash="dash", line_color="black", annotation_text=f"p50 {res['p50']:.2f}ms")
    hist.add_vline(x=res["p95"], line_dash="dot", line_color="#d62728", annotation_text=f"p95 {res['p95']:.2f}ms")
    hist.update_layout(
        title=f"Live single-window inference latency — {variant_choice}",
        xaxis_title="Latency (ms)", yaxis_title="count",
        template="plotly_white", height=360,
    )
    st.plotly_chart(hist, use_container_width=True)
