"""Measure FP32/INT8/ONNX on the frozen optimized folds, without retraining."""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from scipy.special import expit

from src.data.artifacts import load_windows
from src.models.compression import dynamic_quantize, export_onnx
from src.models.search import model_from_params
from src.utils.artifacts import (
    file_digest,
    new_run_dir,
    provenance,
    require_new_directory,
    snapshot_sources,
    write_json,
)
from src.utils.metrics import per_patient_f1, ranking_metrics, threshold_metrics
from src.validation import predict, preprocess_windows


def latency(call, warmup=20, iterations=100):
    for _ in range(warmup):
        call()
    elapsed = []
    for _ in range(iterations):
        start = time.perf_counter()
        call()
        elapsed.append(1000 * (time.perf_counter() - start))
    return {"p50_ms": float(np.median(elapsed)), "p95_ms": float(np.percentile(elapsed, 95))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()
    metadata = json.loads((args.run / "metrics.json").read_text())
    if metadata["protocol"] != "nested_patient_cv_v1":
        raise ValueError("Expérience neuronale incompatible.")
    data = load_windows(args.windows, expected_window=metadata["window"],
                        expected_metadata={"dataset": metadata["dataset"], "stride": 30})
    with np.load(args.run / "oof_scores.npz", allow_pickle=False) as archive:
        oof = {key: archive[key] for key in archive.files}
    if not (np.array_equal(oof["y"], data["y"]) and np.array_equal(oof["groups"], data["patient_id"])):
        raise ValueError("Les fenêtres ne correspondent pas aux scores d'origine.")
    if metadata["provenance"].get("windows_sha256") != file_digest(args.windows):
        raise ValueError("L'archive d'entrée diffère de l'archive d'entraînement.")
    out = require_new_directory(args.out_dir) if args.out_dir else new_run_dir(args.run, "compression")
    snapshot_sources(out / "source_snapshot")
    run_provenance = provenance([args.run / "metrics.json", args.windows])
    torch.set_num_threads(4)
    variants = {key: np.full(len(data["y"]), np.nan, dtype=np.float32)
                for key in ("fp32", "int8_dynamic", "int8_single_window", "onnx_fp32")}
    rows, reload_checks = [], []
    for detail in metadata["folds"]:
        k = detail["fold"]
        indices = np.flatnonzero(oof["fold"] == k)
        X = preprocess_windows(data["X"][indices], detail.get("normalization", "zscore_per_window"))
        model = model_from_params(detail["params"], metadata["window"])
        model.load_state_dict(torch.load(args.run / f"fold{k}.pt", map_location="cpu", weights_only=True))
        model.eval()
        variants["fp32"][indices] = predict(model, X)
        max_delta = float(np.max(np.abs(variants["fp32"][indices] - oof["scores"][indices])))
        if max_delta > 1e-6:
            raise ValueError(f"Les poids du pli {k} ne reproduisent pas ses scores.")
        reload_checks.append({"fold": k, "max_abs_delta": max_delta})
        sample = torch.from_numpy(X[:1])
        with torch.inference_mode():
            measured = latency(lambda m=model, x=sample: m(x))
        rows.append({"fold": k, "variant": "fp32", "size_kib": (args.run / f"fold{k}.pt").stat().st_size / 1024,
                     **measured, "normalization": detail["normalization"]})

        quantized = dynamic_quantize(model)
        variants["int8_dynamic"][indices] = predict(quantized, X)
        path = out / f"fold{k}_int8.pt"
        with path.open("xb") as stream:
            torch.save(quantized.state_dict(), stream)
        # Restore only this locally generated trusted checkpoint, and verify it.
        restored = dynamic_quantize(model)
        restored.load_state_dict(torch.load(path, map_location="cpu", weights_only=False))
        # Dynamic activation quantization can depend on the batch composition.
        # Check reload using exactly the same first batch as the original pass.
        n_check = min(512, len(X))
        np.testing.assert_allclose(predict(restored, X[:n_check]),
                                   variants["int8_dynamic"][indices[:n_check]], atol=1e-7)
        variants["int8_single_window"][indices] = predict(quantized, X, batch_size=1)
        with torch.inference_mode():
            measured = latency(lambda m=quantized, x=sample: m(x))
        rows.append({"fold": k, "variant": "int8_dynamic", "size_kib": path.stat().st_size / 1024,
                     **measured, "normalization": detail["normalization"]})
        rows.append({"fold": k, "variant": "int8_single_window", "size_kib": path.stat().st_size / 1024,
                     **measured, "normalization": detail["normalization"]})

        onnx_path = export_onnx(model, metadata["window"], out / f"fold{k}.onnx")
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        session = ort.InferenceSession(str(onnx_path), sess_options=options, providers=["CPUExecutionProvider"])
        for start in range(0, len(indices), 512):
            logits = session.run(None, {"rr_window": X[start:start + 512]})[0]
            variants["onnx_fp32"][indices[start:start + 512]] = expit(logits)
        np.testing.assert_allclose(variants["onnx_fp32"][indices], variants["fp32"][indices], atol=1e-5)
        measured = latency(lambda s=session, x=X[:1]: s.run(None, {"rr_window": x}))
        rows.append({"fold": k, "variant": "onnx_fp32", "size_kib": onnx_path.stat().st_size / 1024,
                     **measured, "normalization": detail["normalization"]})
        print(f"[compression] pli {k} vérifié", flush=True)

    results = {}
    for name, scores in variants.items():
        if not np.isfinite(scores).all():
            raise ValueError("Scores de compression incomplets.")
        decisions = (scores >= oof["thresholds"]).astype(np.float32)
        subset = [r for r in rows if r["variant"] == name]
        results[name] = {
            "metrics": {**ranking_metrics(data["y"], scores), **threshold_metrics(data["y"], scores),
                        **per_patient_f1(data["y"], scores, data["patient_id"])},
            "calibrated_metrics": {**threshold_metrics(data["y"], decisions),
                                   **per_patient_f1(data["y"], decisions, data["patient_id"])},
            "size_kib_mean": float(np.mean([r["size_kib"] for r in subset])),
            "size_kib_min": min(r["size_kib"] for r in subset),
            "size_kib_max": max(r["size_kib"] for r in subset),
            "ensemble_size_kib": sum(r["size_kib"] for r in subset),
            "latency_p50_ms_mean": float(np.mean([r["p50_ms"] for r in subset])),
        }
    with (out / "scores.npz").open("xb") as stream:
        np.savez(stream, y=data["y"], groups=data["patient_id"], fold=oof["fold"],
                 thresholds=oof["thresholds"], **variants)
    with (out / "measurements.csv").open("x") as stream:
        writer = csv.DictWriter(stream, fieldnames=["fold", "variant", "size_kib", "p50_ms", "p95_ms", "normalization"])
        writer.writeheader()
        writer.writerows(rows)
    write_json(out / "metrics.json", {
        "protocol": "frozen_nested_models_compression_v2", "training_run": str(args.run),
        "variants": results, "folds": metadata["folds"], "reload_checks": reload_checks,
        "provenance": run_provenance,
        "latency_configuration": {"threads": 4, "warmup": 20, "iterations": 100, "batch_size": 1},
        "size_definition": "Serialized weights/graph per fold; ensemble size is the sum of all five models, excluding runtime/RAM.",
        "quality_batch_sizes": {"fp32": 512, "int8_dynamic": 512, "int8_single_window": 1, "onnx_fp32": 512},
        "batch_sensitivity_max_abs_score_delta": float(np.max(np.abs(variants["int8_dynamic"] - variants["int8_single_window"]))),
        "threshold_policy": "Original inner thresholds frozen; no recalibration on outer test patients."})
    print(json.dumps(results, indent=2), flush=True)
    print(out, flush=True)


if __name__ == "__main__":
    main()
