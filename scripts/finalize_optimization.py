"""Build a local, hash-verified candidate manifest after all v2 evaluations finish."""
from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import numpy as np

from scripts.summarize_optimization import compare, load_scores
from src.utils.artifacts import file_digest, require_new_directory, snapshot_sources, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("artifacts/optimization_v2"))
    args = parser.parse_args()
    root = args.root
    external = {}
    for path in sorted((root / "cnn_lstm").glob("evaluation_*/eval_ltafdb.json")):
        meta = json.loads(path.read_text())
        key = f"{meta.get('variant', 'fp32')}_batch{meta.get('runtime', {}).get('batch_size', 512)}"
        external[key] = (path, meta)
    hrv_path = sorted((root / "hrv").glob("evaluation_ltafdb_*/eval_ltafdb.json"))[-1]
    external["hrv_ensemble"] = (hrv_path, json.loads(hrv_path.read_text()))
    required = {"fp32_batch512", "int8_dynamic_batch512", "int8_dynamic_batch1", "onnx_fp32_batch512", "hrv_ensemble"}
    if not required.issubset(external):
        raise ValueError(f"Évaluations manquantes : {required - external.keys()}")
    cases, source_hashes = {}, {}
    external_scores = {}
    for name in sorted(required):
        path, meta = external[name]
        data = load_scores(path.parent / "scores.npz")
        external_scores[name] = data["scores"]
        cases[f"{name}_fixed"] = data, data["scores"], data["scores"] >= .5
        cases[f"{name}_vote"] = data, data["scores"], data["calibrated_decisions"].astype(bool)
        source_hashes[str(path)] = file_digest(path)
        source_hashes[str(path.parent / "scores.npz")] = file_digest(path.parent / "scores.npz")
    stats, _ = compare(cases, [("fp32_batch512_fixed", "hrv_ensemble_fixed"),
                              ("fp32_batch512_vote", "hrv_ensemble_vote"),
                              ("int8_dynamic_batch1_vote", "fp32_batch512_vote")])
    max_onnx_delta = float(np.max(np.abs(external_scores["onnx_fp32_batch512"] - external_scores["fp32_batch512"])))
    if max_onnx_delta > 1e-5:
        raise ValueError("L'export ONNX ne reproduit pas le modèle FP32 sur LTAFDB.")
    stats["onnx_max_abs_score_delta"] = max_onnx_delta
    stats["source_sha256"] = source_hashes
    stats["uncertainty_scope"] = "Patient bootstrap conditional on predictions; no retraining and no new prospective cohort."

    historical_path = Path("artifacts/reliability_v1/historical_manifest.json")
    historical = json.loads(historical_path.read_text())
    changed = [name for name, digest in historical["files"].items() if file_digest(Path(name)) != digest]
    if changed:
        raise ValueError(f"Fichiers historiques modifiés : {changed}")

    training = json.loads((root / "cnn_lstm/metrics.json").read_text())
    compression = json.loads((root / "compression_verified/metrics.json").read_text())
    folds = []
    for detail in training["folds"]:
        k = detail["fold"]
        checkpoints = {"fp32": root / "cnn_lstm" / f"fold{k}.pt",
                       "int8_dynamic": root / "compression_verified" / f"fold{k}_int8.pt",
                       "onnx_fp32": root / "compression_verified" / f"fold{k}.onnx"}
        folds.append({"fold": k, "params": detail["params"], "normalization": detail["normalization"],
                      "threshold": detail["threshold"],
                      "checkpoints": {name: {"path": str(path.relative_to(root)), "sha256": file_digest(path)}
                                      for name, path in checkpoints.items()}})
    out = require_new_directory(root / "release_candidate")
    snapshot_sources(out / "source_snapshot")
    write_json(out / "external_comparison.json", stats)
    write_json(out / "manifest.json", {
        "schema_version": 1, "status": "local_research_candidate",
        "artifact_root_relative_to_manifest": "..", "training_protocol": training["config"]["protocol"],
        "input": {"rr_count": 60, "units": "seconds", "valid_rr_bounds": [.3, 2.],
                  "preprocessing": "Apply the recorded transform for each model before inference."},
        "classes": {"0": "non-AFIB/AFL", "1": "AFIB or AFL majority"},
        "ensemble": {"members": len(folds), "primary_decision": "mean raw scores >= 0.5",
                     "secondary_decision": "majority vote using frozen inner-AFDB thresholds"},
        "folds": folds, "compression_measurements": compression["variants"],
        "external_metrics": {name: meta for name, (_, meta) in external.items()},
        "integrity": {"historical_files_checked": len(historical["files"]), "historical_files_changed": 0},
        "runtime_note": "INT8 scores depend on batch composition; batch=1 was checked on complete AFDB and LTAFDB.",
        "limitations": ["Retrospective public-data evaluation; no prospective clinical validation.",
                        "Weights/graphs sizes exclude inference runtime and RAM.",
                        "Five-model ensemble size differs from the size of a single member."],
    })
    bundle = root / "candidate_bundle.zip"
    curated = []
    for folder in (root / "cnn_lstm", root / "hrv", root / "compression_verified", out):
        curated.extend(p for p in folder.rglob("*") if p.is_file())
    curated.extend(p for folder in root.glob("summary_*") for p in folder.rglob("*") if p.is_file())
    curated.append(root / "plan.json")
    with zipfile.ZipFile(bundle, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(set(curated)):
            archive.write(path, str(path.relative_to(root.parent)))
        archive.write("docs/optimization.md", "optimization_v2/docs/optimization.md")
    print(f"Candidat local créé : {out}")
    print(f"ONNX external maximum score delta: {max_onnx_delta:.3g}")
    print(f"Archive locale : {bundle} ({bundle.stat().st_size / 1024**2:.1f} Mio)")
    print(json.dumps(stats["paired_differences"], indent=2))


if __name__ == "__main__":
    main()
