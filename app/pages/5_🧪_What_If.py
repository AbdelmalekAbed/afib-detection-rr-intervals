"""What-If Lab — sketch an RR series by hand and watch the model react.

The user gets a 60-RR-interval window to play with. Three modes:
  1. Built-in templates (normal sinus / AFib-like / ectopic / atrial flutter)
  2. Per-RR interval sliders for fine-grained edits
  3. Add Gaussian noise / replace random beats with ectopics

The Phase 5 AFDB source model scores the window in real time. Useful for the
jury to see what the model latches on to.
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from app.utils import WINDOW, header, load_source_model, predict_scores

st.set_page_config(page_title="What-If Lab", page_icon="🧪", layout="wide")

header(
    "🧪 What-If Lab",
    f"Build a {WINDOW}-RR window from a template and watch the AFib detector decide. "
    "Edits update the prediction live.",
)

# --------------------------------------------------------------------------- #
# Templates
# --------------------------------------------------------------------------- #
def template_normal_sinus(rng):
    """Regular ~70 BPM with normal RR variability (~30ms SD)."""
    base = 0.85  # ~70 BPM
    rr = base + rng.normal(0, 0.030, size=WINDOW)
    return np.clip(rr, 0.4, 1.6).astype(np.float32)


def template_afib(rng):
    """Highly irregular RR — chaotic, no periodicity."""
    base = 0.75
    rr = base + rng.normal(0, 0.18, size=WINDOW)
    return np.clip(rr, 0.35, 1.4).astype(np.float32)


def template_ectopic_burst(rng):
    """Normal sinus with a short burst of ectopics (compensatory pause + premature beat)."""
    rr = template_normal_sinus(rng).copy()
    burst_start = rng.integers(15, 40)
    for i in range(3):
        if burst_start + 2 * i + 1 >= WINDOW:
            break
        rr[burst_start + 2 * i] = 0.55      # premature
        rr[burst_start + 2 * i + 1] = 1.15  # compensatory pause
    return rr


def template_aflutter(rng):
    """Atrial flutter — regular RR but ~150 BPM and very low variability."""
    base = 0.42  # ~140 BPM
    rr = base + rng.normal(0, 0.012, size=WINDOW)
    return np.clip(rr, 0.30, 0.60).astype(np.float32)


templates = {
    "Normal sinus rhythm (70 BPM, regular)": template_normal_sinus,
    "Atrial fibrillation (chaotic RR)": template_afib,
    "Sinus + ectopic burst": template_ectopic_burst,
    "Atrial flutter (regular, ~140 BPM)": template_aflutter,
}

# --------------------------------------------------------------------------- #
# Controls
# --------------------------------------------------------------------------- #
ctrl1, ctrl2, ctrl3 = st.columns([2, 1, 1])

with ctrl1:
    template_name = st.selectbox("Starting template", list(templates.keys()))

with ctrl2:
    seed = st.number_input("Random seed", 0, 9999, 42, 1)

with ctrl3:
    noise_sd = st.slider("Add Gaussian noise (SD, s)", 0.0, 0.3, 0.0, 0.005)

regen = st.button("🎲 Regenerate", help="Resamples the template at the chosen seed.")

# Use session state so widget edits persist across reruns until regen
if "rr_state" not in st.session_state or regen:
    rng = np.random.default_rng(int(seed))
    rr = templates[template_name](rng)
    if noise_sd > 0:
        rr = rr + rng.normal(0, noise_sd, size=WINDOW)
        rr = np.clip(rr, 0.25, 1.8)
    st.session_state.rr_state = rr.astype(np.float32)

rr = st.session_state.rr_state.copy()

# --------------------------------------------------------------------------- #
# Per-beat editor (advanced)
# --------------------------------------------------------------------------- #
with st.expander("✏️ Edit specific beats", expanded=False):
    st.caption(
        "Each beat (1-60) can be tweaked independently. Useful to inject a single ectopic "
        "or stretch a couple of intervals and see whether the model flips its decision."
    )
    edit_cols = st.columns(6)
    edited = rr.copy()
    for i in range(WINDOW):
        col = edit_cols[i % 6]
        with col:
            edited[i] = st.number_input(
                f"RR[{i}]", value=float(rr[i]),
                min_value=0.25, max_value=1.8, step=0.01, format="%.3f",
                key=f"rr_{i}", label_visibility="collapsed",
            )
    if st.button("Apply beat edits"):
        st.session_state.rr_state = edited.astype(np.float32)
        rr = st.session_state.rr_state.copy()
        st.rerun()

# --------------------------------------------------------------------------- #
# Prediction
# --------------------------------------------------------------------------- #
with st.spinner("Loading source model..."):
    model, best_params = load_source_model()

# z-score the window (same preprocessing as training)
mean = rr.mean()
std = rr.std() + 1e-6
rr_n = ((rr - mean) / std).astype(np.float32)

prob = float(predict_scores(model, rr_n[None, :])[0])
decision = "🔴 AFib" if prob >= 0.5 else "🟢 Normal"

# --------------------------------------------------------------------------- #
# Display
# --------------------------------------------------------------------------- #
left, right = st.columns([3, 1])

with left:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=np.arange(WINDOW), y=rr, mode="lines+markers",
        line=dict(color="#1f77b4", width=2),
        marker=dict(size=7, color=rr, colorscale="RdYlGn_r",
                     line=dict(width=1, color="white"), showscale=False),
        name="RR (s)",
    ))
    fig.add_hline(y=rr.mean(), line_dash="dash", line_color="black", opacity=0.4,
                   annotation_text=f"mean={rr.mean():.3f}s")
    fig.update_layout(
        template="plotly_white", height=380,
        title=f"60 RR intervals — mean {rr.mean()*1000:.0f} ms, SD {rr.std()*1000:.0f} ms ({60/rr.mean():.0f} BPM)",
        xaxis_title="Beat index in window",
        yaxis_title="RR interval (s)",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Poincaré plot
    fig_p = go.Figure()
    fig_p.add_trace(go.Scatter(
        x=rr[:-1], y=rr[1:], mode="markers",
        marker=dict(size=8, color="#d62728", line=dict(width=0.5, color="white"), opacity=0.85),
    ))
    fig_p.update_layout(
        template="plotly_white", height=340,
        title="Poincaré plot — RR(n) vs RR(n+1)",
        xaxis_title="RR(n)", yaxis_title="RR(n+1)",
    )
    fig_p.update_xaxes(scaleanchor="y", scaleratio=1)
    st.plotly_chart(fig_p, use_container_width=True)
    st.caption(
        "Tight cluster = regular (sinus rhythm). Cloud spread out = irregular (AFib). "
        "Two lobes = bigeminy / ectopic pattern. Watch this plot react to your edits."
    )

with right:
    st.markdown("### Verdict")
    st.markdown(f"#### {decision}")
    st.metric("AFib probability", f"{prob:.1%}",
               delta=f"{(prob - 0.5):+.2%} from threshold")

    st.progress(prob)

    st.markdown("---")
    st.markdown("**Window stats**")
    rr_diff = np.diff(rr)
    st.markdown(f"- Heart rate: **{60/rr.mean():.0f} BPM**")
    st.markdown(f"- RR mean: **{rr.mean()*1000:.0f} ms**")
    st.markdown(f"- RR SD: **{rr.std()*1000:.0f} ms**")
    st.markdown(f"- RMSSD: **{np.sqrt((rr_diff**2).mean())*1000:.0f} ms**")
    st.markdown(f"- pNN50: **{(np.abs(rr_diff) > 0.05).mean():.1%}**")

    st.markdown("---")
    st.markdown("**Model used**")
    st.caption(
        f"Phase 5 AFDB source model, architecture Phase 3.5 "
        f"({sum(p.numel() for p in model.parameters()):,} params, w={WINDOW}). "
        "Trained on all of AFDB. F1/patient on LTAFDB zero-shot = 0.713."
    )
