"""Shared helpers for the AFib Sandbox Streamlit app."""
from __future__ import annotations

import io
import json
import sys
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import streamlit as st
import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.cv import zscore_per_window
from src.models.cnn_lstm import CNNLSTM

REPORTS = PROJECT_ROOT / "reports"
FIGURES = REPORTS / "figures"
CKPT = REPORTS / "checkpoints"

WINDOW = 60
STRIDE = 30


# --------------------------------------------------------------------------- #
# Caching
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def load_results_json(name: str) -> dict:
    return json.loads((REPORTS / f"{name}_results.json").read_text())


@st.cache_data(show_spinner=False)
def load_oof_npz(name: str) -> dict:
    path = REPORTS / f"{name}_oof_scores.npz"
    if not path.exists():
        return {}
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.keys()}


@st.cache_data(show_spinner=False)
def load_compression_csv(prefix: str):
    import pandas as pd
    return pd.read_csv(REPORTS / f"{prefix}_compression.csv")


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def make_model(best: dict, input_length: int = WINDOW) -> CNNLSTM:
    if best["n_cnn_blocks"] == 2:
        channels = (best["cnn_first"], best["cnn_first"] * 2)
    else:
        channels = (best["cnn_first"], best["cnn_first"] * 2, best["cnn_first"] * 4)
    return CNNLSTM(
        input_length=input_length,
        cnn_channels=channels,
        kernel_size=best["kernel_size"],
        pool_size=2,
        cnn_dropout=best["cnn_dropout"],
        batch_norm=True,
        lstm_hidden=best["lstm_hidden"],
        lstm_layers=best["lstm_layers"],
        bidirectional=True,
        lstm_dropout=0.2 if best["lstm_layers"] > 1 else 0.0,
        head_hidden=best["head_hidden"],
        head_dropout=best["head_dropout"],
    )


@st.cache_resource(show_spinner=False)
def load_source_model() -> tuple[nn.Module, dict]:
    """Load the Phase 5 AFDB source model — the one trained on all AFDB
    that generalizes to LTAFDB zero-shot at F1/patient=0.71."""
    raw = json.loads((REPORTS / "phase35_best_params.json").read_text())
    best = raw["best_params"] if "best_params" in raw else raw
    model = make_model(best, WINDOW)
    state = torch.load(CKPT / "phase5_source_afdb.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model, best


@st.cache_resource(show_spinner=False)
def load_variant_model(variant: str) -> nn.Module:
    """Load a Phase 4 compression variant (archi 100k). Variant name is the
    suffix of fold0_<variant>.pt in reports/checkpoints/."""
    # Use Phase 3 best params for archi 100k variants
    raw = json.loads((REPORTS / "phase3_results.json").read_text())
    best = raw["best_params"]
    model = make_model(best, WINDOW)
    path = CKPT / f"fold0_{variant}.pt"
    state = torch.load(path, map_location="cpu", weights_only=False)
    if variant == "int8_dynamic" or variant.endswith("_int8"):
        # Need to apply quantization before loading the quantized state
        model = torch.quantization.quantize_dynamic(
            model, qconfig_spec={nn.Linear, nn.LSTM}, dtype=torch.qint8,
        )
    model.load_state_dict(state)
    model.eval()
    return model


# --------------------------------------------------------------------------- #
# Inference helpers
# --------------------------------------------------------------------------- #
def predict_scores(model: nn.Module, X: np.ndarray, batch_size: int = 512) -> np.ndarray:
    """Predict AFib probability for a batch of windows."""
    model.eval()
    out = np.empty(len(X), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.from_numpy(X[i : i + batch_size]).float()
            out[i : i + batch_size] = torch.sigmoid(model(xb)).cpu().numpy()
    return out


def benchmark_latency(model: nn.Module, n_iters: int = 100, warmup: int = 20) -> dict:
    """Single-window CPU latency benchmark."""
    model.eval()
    x = torch.randn(1, WINDOW)
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        ts = []
        for _ in range(n_iters):
            t0 = time.perf_counter()
            model(x)
            ts.append((time.perf_counter() - t0) * 1000)
    return {
        "p50": float(np.median(ts)),
        "p95": float(np.percentile(ts, 95)),
        "mean": float(np.mean(ts)),
        "raw_ms": ts,
    }


def state_dict_size_kb(model: nn.Module) -> float:
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return len(buf.getvalue()) / 1024.0


# --------------------------------------------------------------------------- #
# Windowing
# --------------------------------------------------------------------------- #
def slide_windows(rr: np.ndarray, window: int = WINDOW, stride: int = STRIDE):
    """Sliding windows + their start indices."""
    n = len(rr)
    if n < window:
        return np.empty((0, window), dtype=np.float32), np.empty(0, dtype=np.int64)
    starts = np.arange(0, n - window + 1, stride, dtype=np.int64)
    X = np.stack([rr[s : s + window] for s in starts]).astype(np.float32)
    return X, starts


def per_patient_summary(y: np.ndarray, scores: np.ndarray, groups: np.ndarray, threshold: float = 0.5):
    """Per-patient F1, sensitivity, specificity, support."""
    from sklearn.metrics import f1_score
    rows = []
    for pid in np.unique(groups):
        m = groups == pid
        if m.sum() == 0:
            continue
        n_pos = int(y[m].sum())
        n_neg = int(m.sum() - n_pos)
        if n_pos == 0 or n_neg == 0:
            rows.append({"patient": str(pid), "f1": float("nan"), "afib_rate": n_pos / m.sum(),
                          "n_windows": int(m.sum()), "scorable": False})
            continue
        y_pred = (scores[m] >= threshold).astype(int)
        rows.append({
            "patient": str(pid),
            "f1": float(f1_score(y[m], y_pred, zero_division=0)),
            "afib_rate": n_pos / m.sum(),
            "n_windows": int(m.sum()),
            "scorable": True,
        })
    return rows


# --------------------------------------------------------------------------- #
# Sample data — built-in RR series for the live demo
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def get_demo_patients_ltafdb():
    """Build a {patient_id: {rr: np.array, y: np.array, scores: dict}} from Phase 5 OOF.
    We don't have raw RR series stored, only the scored windows. So we reconstruct what
    we can show: y / scores aligned at window granularity, indexed by patient.
    """
    oof = load_oof_npz("phase5")
    if not oof:
        return {}
    groups = oof["groups_ltafdb"]
    y = oof["y_ltafdb"]
    patients = {}
    for pid in np.unique(groups):
        m = groups == pid
        patients[str(pid)] = {
            "y": y[m].astype(int),
            "zero_shot": oof["zero_shot"][m],
            "scratch": oof["scratch"][m],
            "finetuned": oof["finetuned"][m],
            "fold": int(oof["fold"][m][0]) if "fold" in oof else -1,
            "n_windows": int(m.sum()),
            "afib_rate": float(y[m].mean()),
        }
    return patients


@st.cache_data(show_spinner=False)
def get_demo_patients_afdb():
    """AFDB patient breakdown from Phase 4 OOF scores."""
    oof = load_oof_npz("phase4")
    if not oof:
        return {}
    groups = oof["groups"]
    y = oof["y"]
    patients = {}
    variant_keys = [k for k in oof.keys() if k not in {"y", "groups", "fold"}]
    for pid in np.unique(groups):
        m = groups == pid
        rec = {
            "y": y[m].astype(int),
            "fold": int(oof["fold"][m][0]) if "fold" in oof else -1,
            "n_windows": int(m.sum()),
            "afib_rate": float(y[m].mean()),
        }
        for k in variant_keys:
            rec[k] = oof[k][m]
        patients[str(pid)] = rec
    return patients


# --------------------------------------------------------------------------- #
# UI helpers
# --------------------------------------------------------------------------- #
def header(title: str, subtitle: str = ""):
    st.markdown(f"## {title}")
    if subtitle:
        st.caption(subtitle)


def kpi_row(items: list[tuple[str, str, str | None]]):
    """Render N KPI cards in a row. Each item = (label, value, delta or None)."""
    cols = st.columns(len(items))
    for col, (label, value, delta) in zip(cols, items):
        with col:
            st.metric(label, value, delta)
