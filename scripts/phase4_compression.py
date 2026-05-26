"""Phase 4 — Compression du meilleur CNN-LSTM Phase 3 (w=60 default).

Pipeline :
  1. Reconstruire les fenêtres AFDB w=60 / stride=30 (mêmes paramètres que l'ablation Phase 3).
  2. Entraîner 5-fold OOF patient-level avec les ``best_params`` Phase 3 → 5 checkpoints FP32.
  3. Pour chaque variante de compression, ré-évaluer OOF + mesurer taille fichier et latence
     single-window CPU.
  4. Sauvegarder ``reports/phase4_results.json``, ``reports/phase4_compression.csv``,
     ``reports/figures/16_phase4_*.png``, et un ``reports/checkpoints/`` avec un modèle
     par variante (depuis le fold 0).

Variantes mesurées :
  - fp32                  : référence
  - int8_dynamic          : quantization dynamique (Linear + LSTM)
  - prune30 / prune50 / prune70 : magnitude pruning unstructured global (Conv1d + Linear)
  - prune50_int8          : pruning 50% + quantization
  - onnx_fp32             : export ONNX + inférence onnxruntime
"""
from __future__ import annotations

import argparse
import copy
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn.utils import prune

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.cv import zscore_per_window
from torch.utils.data import DataLoader, TensorDataset
from torch import nn as _nn
from sklearn.metrics import roc_auc_score
from src.data.loader import list_records, load_record_metadata
from src.data.rr_extract import clean_rr_series, extract_rr_series
from src.data.windowing import build_windowed_dataset
from src.models.cnn_lstm import CNNLSTM
from src.utils.metrics import per_patient_f1, ranking_metrics, threshold_metrics
from src.utils.seed import set_seed
from src.utils.splits import patient_kfold

WINDOW = 60
STRIDE = 30
SEED = 42
N_SPLITS = 5
EPOCHS = 12
BATCH_SIZE = 512
PATIENCE = 3
LATENCY_WARMUP = 50
LATENCY_ITERS = 200


def train_one_fold_keep_best(
    model: _nn.Module,
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_va: np.ndarray, y_va: np.ndarray,
    epochs: int, batch_size: int, lr: float, patience: int,
    pos_weight: float, seed: int,
) -> tuple[dict, np.ndarray]:
    """Train and return (best_state_dict, best_val_scores) from the lowest-val-loss epoch."""
    torch.manual_seed(seed)
    Xt = torch.from_numpy(X_tr).float()
    yt = torch.from_numpy(y_tr.astype(np.float32))
    Xv = torch.from_numpy(X_va).float()
    yv = torch.from_numpy(y_va.astype(np.float32))
    loader = DataLoader(TensorDataset(Xt, yt), batch_size=batch_size, shuffle=True)
    optim = torch.optim.AdamW(model.parameters(), lr=lr)
    pw = torch.tensor([pos_weight], dtype=torch.float32)
    crit = _nn.BCEWithLogitsLoss(pos_weight=pw)

    best_loss = float("inf")
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    best_scores = None
    bad = 0
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            optim.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            optim.step()
        model.eval()
        with torch.no_grad():
            val_logits = model(Xv)
            val_loss = float(crit(val_logits, yv).item())
            scores = torch.sigmoid(val_logits).cpu().numpy()
        if val_loss < best_loss - 1e-4:
            best_loss = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            best_scores = scores
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_scores is None:
        best_scores = scores
    return best_state, best_scores


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def build_afdb_windows(window: int, stride: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dataset_dir = PROJECT_ROOT / "data" / "raw" / "afdb"
    series_by_patient = {}
    for rec in list_records(dataset_dir):
        try:
            meta = load_record_metadata(dataset_dir, rec)
            s = clean_rr_series(extract_rr_series(meta))
            if len(s.rr_seconds) >= window:
                series_by_patient[rec] = s
        except Exception:
            continue
    w = build_windowed_dataset(series_by_patient, window_size=window, stride=stride, label_strategy="majority")
    return w.X.astype(np.float32), w.y.astype(np.int8), np.asarray(w.patient_id)


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def make_model(best: dict, input_length: int) -> CNNLSTM:
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


# --------------------------------------------------------------------------- #
# Compression primitives
# --------------------------------------------------------------------------- #
def quantize_dynamic_int8(model: nn.Module) -> nn.Module:
    model.eval()
    return torch.quantization.quantize_dynamic(
        model,
        qconfig_spec={nn.Linear, nn.LSTM},
        dtype=torch.qint8,
    )


def prune_global(model: nn.Module, amount: float, make_permanent: bool = True) -> nn.Module:
    """Global L1 unstructured pruning on Conv1d + Linear weights.

    If ``make_permanent`` is False the pruning re-parametrization (mask) stays
    attached — required to *keep* sparsity through subsequent fine-tuning.
    """
    targets = []
    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            targets.append((module, "weight"))
    prune.global_unstructured(targets, pruning_method=prune.L1Unstructured, amount=amount)
    if make_permanent:
        for module, name in targets:
            prune.remove(module, name)
    return model


def make_pruning_permanent(model: nn.Module) -> nn.Module:
    """Remove all pruning reparametrizations, freezing zeros into the weights."""
    for module in model.modules():
        for name in ("weight",):
            if hasattr(module, name + "_mask"):
                prune.remove(module, name)
    return model


def measured_sparsity(model: nn.Module) -> float:
    total, zero = 0, 0
    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            w = module.weight
            total += w.numel()
            zero += int((w == 0).sum().item())
    return zero / total if total else 0.0


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def predict_scores(model: nn.Module, X: np.ndarray, batch_size: int = 512) -> np.ndarray:
    model.eval()
    out = np.empty(len(X), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.from_numpy(X[i : i + batch_size]).float()
            out[i : i + batch_size] = torch.sigmoid(model(xb)).cpu().numpy()
    return out


def metric_block(y: np.ndarray, scores: np.ndarray, groups: np.ndarray) -> dict[str, float]:
    return {
        **ranking_metrics(y, scores),
        **threshold_metrics(y, scores, threshold=0.5),
        **per_patient_f1(y, scores, groups, threshold=0.5),
    }


# --------------------------------------------------------------------------- #
# Size + latency benchmarks
# --------------------------------------------------------------------------- #
def state_dict_size_kb(model: nn.Module) -> float:
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return len(buf.getvalue()) / 1024.0


def file_size_kb(path: Path) -> float:
    return path.stat().st_size / 1024.0


def benchmark_latency_torch(model: nn.Module, input_length: int) -> dict[str, float]:
    """Single-window CPU inference latency."""
    model.eval()
    x = torch.randn(1, input_length)
    with torch.no_grad():
        for _ in range(LATENCY_WARMUP):
            model(x)
        ts = []
        for _ in range(LATENCY_ITERS):
            t0 = time.perf_counter()
            model(x)
            ts.append((time.perf_counter() - t0) * 1000.0)  # ms
    return {
        "latency_ms_p50": float(np.median(ts)),
        "latency_ms_p95": float(np.percentile(ts, 95)),
        "latency_ms_mean": float(np.mean(ts)),
    }


def benchmark_latency_onnx(sess, input_length: int, input_name: str) -> dict[str, float]:
    x = np.random.randn(1, input_length).astype(np.float32)
    for _ in range(LATENCY_WARMUP):
        sess.run(None, {input_name: x})
    ts = []
    for _ in range(LATENCY_ITERS):
        t0 = time.perf_counter()
        sess.run(None, {input_name: x})
        ts.append((time.perf_counter() - t0) * 1000.0)
    return {
        "latency_ms_p50": float(np.median(ts)),
        "latency_ms_p95": float(np.percentile(ts, 95)),
        "latency_ms_mean": float(np.mean(ts)),
    }


# --------------------------------------------------------------------------- #
# Main pipeline
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-splits", type=int, default=N_SPLITS)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "reports")
    ap.add_argument("--ckpt-dir", type=Path, default=PROJECT_ROOT / "reports" / "checkpoints")
    ap.add_argument(
        "--params-json",
        type=Path,
        default=PROJECT_ROOT / "reports" / "phase3_results.json",
        help="Path to a JSON file with a 'best_params' key, or the params dict directly.",
    )
    ap.add_argument(
        "--out-prefix",
        type=str,
        default="phase4",
        help="Filename prefix for reports/{prefix}_results.json, _compression.csv, _oof_scores.npz.",
    )
    ap.add_argument(
        "--fig-prefix",
        type=str,
        default="16_phase4",
        help="Filename prefix for reports/figures/{prefix}_*.png.",
    )
    ap.add_argument(
        "--ckpt-tag",
        type=str,
        default="",
        help="Optional tag appended to checkpoint filenames to avoid overwriting Phase 4 ckpts.",
    )
    args = ap.parse_args()

    set_seed(SEED)
    torch.set_num_threads(4)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = args.out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 1. Data
    t0 = time.time()
    X, y, groups = build_afdb_windows(WINDOW, STRIDE)
    print(f"[data] AFDB w={WINDOW}: X={X.shape}, AFib rate={y.mean():.3f}, "
          f"patients={len(np.unique(groups))} ({time.time()-t0:.1f}s)")
    Xn = zscore_per_window(X)

    # 2. Load best params (from --params-json; supports either {"best_params": {...}} or {...})
    raw_params = json.loads(args.params_json.read_text())
    best = raw_params["best_params"] if "best_params" in raw_params else raw_params
    print(f"[params] loaded from {args.params_json} (input_length={WINDOW}): {best}")
    n_params = sum(p.numel() for p in make_model(best, WINDOW).parameters())
    print(f"[params] model parameter count: {n_params:,}")

    # 3. 5-fold OOF training — keep one state_dict per fold
    fold_states: list[dict] = []
    oof_fp32 = np.full(len(y), np.nan, dtype=np.float32)
    fold_arr = np.full(len(y), -1, dtype=np.int8)
    print(f"[train] 5-fold OOF FP32 — epochs={args.epochs}, batch={BATCH_SIZE}, patience={PATIENCE}")
    for k, (tr, va) in enumerate(patient_kfold(groups, y, n_splits=args.n_splits)):
        t0 = time.time()
        n_pos = max(int(y[tr].sum()), 1)
        n_neg = max(int(len(y[tr]) - n_pos), 1)
        pw = n_neg / n_pos
        model = make_model(best, WINDOW)
        best_state, best_scores = train_one_fold_keep_best(
            model=model,
            X_tr=Xn[tr], y_tr=y[tr], X_va=Xn[va], y_va=y[va],
            epochs=args.epochs, batch_size=BATCH_SIZE, lr=best["lr"],
            patience=PATIENCE, pos_weight=pw, seed=SEED + k,
        )
        oof_fp32[va] = best_scores
        fold_arr[va] = k
        fold_states.append(best_state)
        print(f"  fold {k}: trained in {time.time()-t0:.1f}s, "
              f"val_size={len(va)}, scored_pos={int(y[va].sum())}")

    metrics_fp32 = metric_block(y, oof_fp32, groups)
    print(f"[fp32] AUROC={metrics_fp32['auroc']:.4f} F1={metrics_fp32['f1']:.4f} "
          f"F1/patient={metrics_fp32['mean_patient_f1']:.4f}")

    # 4. Helper : run a compression recipe on each fold, gather OOF + size + latency
    def run_variant(name: str, transform) -> dict:
        """``transform(model, fold_idx, tr_idx, va_idx)`` returns the compressed model."""
        t0 = time.time()
        oof = np.full(len(y), np.nan, dtype=np.float32)
        sizes_kb, lat_p50, lat_p95, lat_mean = [], [], [], []
        sparsities = []
        first_compressed = None
        for k, (tr, va) in enumerate(patient_kfold(groups, y, n_splits=args.n_splits)):
            model = make_model(best, WINDOW)
            model.load_state_dict(fold_states[k])
            model = transform(model, k, tr, va)
            oof[va] = predict_scores(model, Xn[va])
            sizes_kb.append(state_dict_size_kb(model))
            lat = benchmark_latency_torch(model, WINDOW)
            lat_p50.append(lat["latency_ms_p50"])
            lat_p95.append(lat["latency_ms_p95"])
            lat_mean.append(lat["latency_ms_mean"])
            sparsities.append(measured_sparsity(model))
            if k == 0:
                first_compressed = model
        m = metric_block(y, oof, groups)
        out = {
            "name": name,
            **m,
            "size_kb_mean": float(np.mean(sizes_kb)),
            "latency_ms_p50": float(np.mean(lat_p50)),
            "latency_ms_p95": float(np.mean(lat_p95)),
            "latency_ms_mean": float(np.mean(lat_mean)),
            "sparsity": float(np.mean(sparsities)),
            "elapsed_s": float(time.time() - t0),
        }
        print(f"[{name}] size={out['size_kb_mean']:.1f}KB latency_p50={out['latency_ms_p50']:.2f}ms "
              f"sparsity={out['sparsity']:.2f} AUROC={out['auroc']:.4f} "
              f"F1/patient={out['mean_patient_f1']:.4f} ({out['elapsed_s']:.1f}s)")
        return out, first_compressed, oof

    variants: dict[str, dict] = {}
    oof_table: dict[str, np.ndarray] = {"fp32": oof_fp32}

    # fp32 baseline measurements (size/latency) — same loop as variants for apples-to-apples
    def t_fp32(m, _k, _tr, _va):
        return m
    tag = f"_{args.ckpt_tag}" if args.ckpt_tag else ""
    v, ckpt_fp32, _ = run_variant("fp32", t_fp32)
    variants["fp32"] = v
    torch.save(ckpt_fp32.state_dict(), args.ckpt_dir / f"fold0_fp32{tag}.pt")

    # int8 dynamic
    def t_int8(m, _k, _tr, _va):
        return quantize_dynamic_int8(m)
    v, ckpt_int8, oof_table["int8_dynamic"] = run_variant("int8_dynamic", t_int8)
    variants["int8_dynamic"] = v
    torch.save(ckpt_int8.state_dict(), args.ckpt_dir / f"fold0_int8_dynamic{tag}.pt")

    # pruning sweep
    for amount in (0.3, 0.5, 0.7):
        key = f"prune{int(amount*100)}"
        def t_prune(m, _k, _tr, _va, _a=amount):
            return prune_global(m, _a)
        v, ckpt, oof_table[key] = run_variant(key, t_prune)
        variants[key] = v
        torch.save(ckpt.state_dict(), args.ckpt_dir / f"fold0_{key}{tag}.pt")

    # prune50 + int8
    def t_prune_int8(m, _k, _tr, _va):
        return quantize_dynamic_int8(prune_global(m, 0.5))
    v, ckpt_p50_int8, oof_table["prune50_int8"] = run_variant("prune50_int8", t_prune_int8)
    variants["prune50_int8"] = v
    torch.save(ckpt_p50_int8.state_dict(), args.ckpt_dir / f"fold0_prune50_int8{tag}.pt")

    # prune70 + short fine-tune (3 epochs) — keep mask active so sparsity is preserved
    FT_EPOCHS = 3
    def t_prune70_ft(m, k_idx, tr, va):
        m = prune_global(m, 0.7, make_permanent=False)  # mask stays attached
        n_pos_tr = max(int(y[tr].sum()), 1)
        n_neg_tr = max(int(len(y[tr]) - n_pos_tr), 1)
        pw_tr = n_neg_tr / n_pos_tr
        Xt = torch.from_numpy(Xn[tr]).float()
        yt = torch.from_numpy(y[tr].astype(np.float32))
        loader = DataLoader(TensorDataset(Xt, yt), batch_size=BATCH_SIZE, shuffle=True)
        optim = torch.optim.AdamW(m.parameters(), lr=best["lr"] * 0.3)
        crit = _nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pw_tr], dtype=torch.float32))
        torch.manual_seed(SEED + 100 + k_idx)
        m.train()
        for _ in range(FT_EPOCHS):
            for xb, yb in loader:
                optim.zero_grad()
                crit(m(xb), yb).backward()
                optim.step()
        return make_pruning_permanent(m)
    v, ckpt_p70_ft, oof_table["prune70_finetune"] = run_variant("prune70_finetune", t_prune70_ft)
    variants["prune70_finetune"] = v
    torch.save(ckpt_p70_ft.state_dict(), args.ckpt_dir / f"fold0_prune70_finetune{tag}.pt")

    # 5. ONNX export + onnxruntime OOF evaluation (export each fold, eval its val split)
    import onnxruntime as ort
    print("[onnx] exporting 5 fold FP32 models + running OOF inference via onnxruntime")
    so = ort.SessionOptions()
    so.intra_op_num_threads = 4
    onnx_oof = np.full(len(y), np.nan, dtype=np.float32)
    onnx_sizes, onnx_lat_p50, onnx_lat_p95, onnx_lat_mean = [], [], [], []
    onnx_max_abs = []
    onnx_t0 = time.time()
    for k, (tr, va) in enumerate(patient_kfold(groups, y, n_splits=args.n_splits)):
        m = make_model(best, WINDOW)
        m.load_state_dict(fold_states[k])
        m.eval()
        onnx_path = args.ckpt_dir / f"fold{k}_fp32{tag}.onnx"
        dummy = torch.zeros(1, WINDOW)
        torch.onnx.export(
            m, dummy, str(onnx_path),
            input_names=["rr_window"], output_names=["logit"],
            dynamic_axes={"rr_window": {0: "batch"}, "logit": {0: "batch"}},
            opset_version=17, external_data=False,
        )
        onnx_sizes.append(file_size_kb(onnx_path))
        sess = ort.InferenceSession(str(onnx_path), sess_options=so, providers=["CPUExecutionProvider"])
        input_name = sess.get_inputs()[0].name

        # Validate equivalence on a small sample of this fold's validation set
        x_check = Xn[va[:64]].astype(np.float32)
        with torch.no_grad():
            torch_scores = torch.sigmoid(m(torch.from_numpy(x_check))).numpy()
        onnx_logits = sess.run(None, {input_name: x_check})[0]
        onnx_scores_check = 1.0 / (1.0 + np.exp(-onnx_logits.squeeze()))
        onnx_max_abs.append(float(np.max(np.abs(torch_scores - onnx_scores_check))))

        # Full OOF inference for this fold's validation split
        scores_va = np.empty(len(va), dtype=np.float32)
        bs = 512
        for i in range(0, len(va), bs):
            xb = Xn[va[i : i + bs]].astype(np.float32)
            logits = sess.run(None, {input_name: xb})[0].squeeze()
            scores_va[i : i + bs] = 1.0 / (1.0 + np.exp(-logits))
        onnx_oof[va] = scores_va

        # latency
        lat = benchmark_latency_onnx(sess, WINDOW, input_name)
        onnx_lat_p50.append(lat["latency_ms_p50"])
        onnx_lat_p95.append(lat["latency_ms_p95"])
        onnx_lat_mean.append(lat["latency_ms_mean"])

    onnx_metrics = metric_block(y, onnx_oof, groups)
    variants["onnx_fp32"] = {
        "name": "onnx_fp32",
        **onnx_metrics,
        "size_kb_mean": float(np.mean(onnx_sizes)),
        "latency_ms_p50": float(np.mean(onnx_lat_p50)),
        "latency_ms_p95": float(np.mean(onnx_lat_p95)),
        "latency_ms_mean": float(np.mean(onnx_lat_mean)),
        "sparsity": 0.0,
        "elapsed_s": float(time.time() - onnx_t0),
        "onnx_max_abs_vs_torch_mean": float(np.mean(onnx_max_abs)),
    }
    oof_table["onnx_fp32"] = onnx_oof
    print(f"[onnx_fp32] size={variants['onnx_fp32']['size_kb_mean']:.1f}KB "
          f"latency_p50={variants['onnx_fp32']['latency_ms_p50']:.2f}ms "
          f"AUROC={onnx_metrics['auroc']:.4f} F1/patient={onnx_metrics['mean_patient_f1']:.4f} "
          f"max|Δ vs torch|={variants['onnx_fp32']['onnx_max_abs_vs_torch_mean']:.2e}")

    # 6. Persist results
    out_json = {
        "config": {
            "window": WINDOW, "stride": STRIDE, "n_splits": args.n_splits,
            "epochs": args.epochs, "batch_size": BATCH_SIZE, "patience": PATIENCE,
            "seed": SEED, "latency_warmup": LATENCY_WARMUP, "latency_iters": LATENCY_ITERS,
            "torch_threads": torch.get_num_threads(),
        },
        "best_params": best,
        "n_params": int(n_params),
        "data": {"n_samples": int(len(y)), "n_patients": int(len(np.unique(groups))),
                 "afib_rate": float(y.mean())},
        "variants": variants,
        "targets_plan": {"size_kb_max": 200.0, "latency_ms_max": 50.0,
                          "f1_patient_min": 0.95, "auroc_min": 0.98},
    }
    (args.out_dir / f"{args.out_prefix}_results.json").write_text(json.dumps(out_json, indent=2))
    print(f"[save] reports/{args.out_prefix}_results.json")

    # CSV
    import pandas as pd
    rows = []
    for name, v in variants.items():
        rows.append({
            "variant": name,
            "size_kb": v["size_kb_mean"],
            "size_ratio_vs_fp32": v["size_kb_mean"] / variants["fp32"]["size_kb_mean"],
            "latency_ms_p50": v["latency_ms_p50"],
            "latency_ms_p95": v["latency_ms_p95"],
            "sparsity": v["sparsity"],
            "auroc": v.get("auroc"),
            "auprc": v.get("auprc"),
            "f1": v.get("f1"),
            "sensitivity": v.get("sensitivity"),
            "specificity": v.get("specificity"),
            "mean_patient_f1": v.get("mean_patient_f1"),
        })
    df = pd.DataFrame(rows)
    df.to_csv(args.out_dir / f"{args.out_prefix}_compression.csv", index=False)
    print(f"[save] reports/{args.out_prefix}_compression.csv\n{df.to_string(index=False)}")

    # Save OOF scores for downstream notebook analysis
    np.savez(
        args.out_dir / f"{args.out_prefix}_oof_scores.npz",
        y=y.astype(np.int8), groups=groups.astype(str), fold=fold_arr,
        **{k: v.astype(np.float32) for k, v in oof_table.items()},
    )
    print(f"[save] reports/{args.out_prefix}_oof_scores.npz")

    # Figures
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df_plot = df.dropna(subset=["mean_patient_f1"]).copy()
    df_plot = df_plot.sort_values("size_kb")
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(df_plot["size_kb"], df_plot["mean_patient_f1"], s=80, color="#1f77b4")
    for _, r in df_plot.iterrows():
        ax.annotate(r["variant"], (r["size_kb"], r["mean_patient_f1"]),
                    xytext=(6, 4), textcoords="offset points", fontsize=9)
    ax.axvline(200, color="#d62728", ls="--", lw=1, label="cible plan : 200 KB")
    ax.set_xlabel("Taille du modèle (KB, state_dict sérialisé)")
    ax.set_ylabel("F1 moyen par patient (5-fold OOF)")
    ax.set_title("Phase 4 — Compression : taille vs qualité")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(fig_dir / f"{args.fig_prefix}_size_vs_f1.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(df["latency_ms_p50"], df["size_kb"], s=80, color="#2ca02c")
    for _, r in df.iterrows():
        ax.annotate(r["variant"], (r["latency_ms_p50"], r["size_kb"]),
                    xytext=(6, 4), textcoords="offset points", fontsize=9)
    ax.axvline(50, color="#d62728", ls="--", lw=1, label="cible plan : 50 ms")
    ax.axhline(200, color="#d62728", ls=":", lw=1)
    ax.set_xlabel("Latence single-window (ms, p50)")
    ax.set_ylabel("Taille (KB)")
    ax.set_title("Phase 4 — Latence vs taille (toutes variantes)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(fig_dir / f"{args.fig_prefix}_latency_vs_size.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(df_plot["variant"], df_plot["latency_ms_p50"],
                  yerr=(df_plot["latency_ms_p95"] - df_plot["latency_ms_p50"]),
                  color="#9467bd", capsize=4)
    ax.axhline(50, color="#d62728", ls="--", lw=1, label="cible 50 ms")
    ax.set_ylabel("Latence (ms, p50 ± p95)")
    ax.set_title("Phase 4 — Latence single-window CPU par variante")
    ax.legend()
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(fig_dir / f"{args.fig_prefix}_latency_bar.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"[save] figures in {fig_dir}/{args.fig_prefix}_*.png")


if __name__ == "__main__":
    main()
